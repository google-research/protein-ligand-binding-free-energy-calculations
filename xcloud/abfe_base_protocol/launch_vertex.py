# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Launch script to run multiple ABFE calculations on GCP.

The --in_path argument specifies where all the MDP, TOP and PDB/GRO 
files are and expects a specific directory structure:

```
path-to-input/
  mdp_files/
    em_l0.mdp
    em_l1.mdp
    eq_l0.mdp
    eq_l1.mdp
    eq_posre_l0.mdp
    eq_posre_l1.mdp
    ti_l0.mdp
    ti_l1.mdp
  struct_top_files/
    ligand1/
      ligand/
        MOL.itp
        ffMOL.itp
        lig.pdb
      protein/
        protein.itp
        protein.pdb
        posre_protein.itp
    ...
    ligandN/
    apo_protein     # optional, if wanting != protein structures for apo and holo simulations
```

Usage:

Prerequisites:
1) gcloud auth login
2) gcloud auth configure-docker

xmanager launch launch_multi_exp.py -- \
  --xm_resource_alloc="user:xcloud/${USER}" \
  --in_path="/gcs/path-to-inputs" \
  --out_path="/gcs/output-folder/" \
  --lig_ids=1,10 \
  --noxm_gpu_sharing_opt_in
"""

from typing import List
import os

from absl import app
from absl import flags
from absl import logging

from xmanager import xm
from xmanager import xm_local


_EXP_NAME = flags.DEFINE_string('exp_name', 'abfe', 
                                'Name of the experiment.', 
                                short_name='n')
_BASE_IMAGE = flags.DEFINE_string('base_image', None, 
                                  'Docker image with Gromacs.', 
                                  short_name='i')
_TOP_PATH = flags.DEFINE_string('top_path', None, 'Path to folder with GRO/TOP files. We expect this directory will have '
                                'one subfolder for each calculation, with the inputs for different ligands, and '
                                'optionally for the apo protein simulations.',
                                short_name='t')
_MDP_PATH = flags.DEFINE_string('mdp_path', None, 'Path to MDP files.', short_name='m')
_OUT_PATH = flags.DEFINE_string('out_path', None, 'Output directory for all calculations (in /gcs/bucket/directory format).',
                                short_name='o')
_LIG_IDS = flags.DEFINE_string('lig_ids', None, 'Ligand IDs for which to run ABFE calculations. '
                               'This can be a comma-separated list (e.g., "0,1,4") or an inclusive range (e.g., "0-9"). '
                               'We expect the folders {top_path}/ligand_{lig_id} to exists.')
_APO_DIR = flags.DEFINE_string('apo_dir', 'apo_protein', 'Name of directory with TOP/GRO files for apo protein simulations.')
_EQUIL_TIME = flags.DEFINE_float('equil_time', 1000.1, 'Equilibration time (in ps) to discard from '
                                 'the equilibrium simulations.')
_NUM_REPEATS = flags.DEFINE_integer('num_repeats', 5,
                                'The number of simulation repeats.')
_TARGET_PRECISION_WAT = flags.DEFINE_float('target_precision_wat', 0., 'Target precision of the dG estimate '
                                           'for the ligand in water calculations. The precision is measured as the '
                                           'standard error in kJ/mol. '
                                           'An early stopping mechanism will be used in which '
                                           'we stop running non-equilibrium simulations once this precision '
                                           'has been achieved. Default is 0 kJ/mol. (i.e. do not use early stopping).')
_TARGET_PRECISION_PRO = flags.DEFINE_float('target_precision_pro', 0., 'Target precision of the dG estimate '
                                           'for the protein in water calculations. The precision is measured as the '
                                           'standard error in kJ/mol. '
                                           'An early stopping mechanism will be used in which '
                                           'we stop running non-equilibrium simulations once this precision '
                                           'has been achieved. Default is 0 kJ/mol. (i.e. do not use early stopping).')
_FREQ_PRECISION_EVAL = flags.DEFINE_integer('freq_precision_eval', 10, 'Frequency with which to evaluate '
                                            'the precision of the dG estimates. The number specified '
                                            'corresponds to how many non-equilibrium transitions are run '
                                            'before estimating free energy differences again.')
_PATIENCE = flags.DEFINE_integer('patience', 3, 'Number of consecutive iterations in which the `target_precision` '
                                 'is reached that are allowed before terminating further non-equilibrium simulations.')
_MIN_NUM_TRANSITIONS = flags.DEFINE_integer('min_num_transitions', 0, 'Minimum number of non-equilibrium transitions '
                                            'we guarantee will be run while using early stopping. This is the number '
                                            'per each individual repeat; e.g., if this is set to 10, 10 transitions will '
                                            'be run for all repeats in state A and all in state B.')
_GPU_TYPE = flags.DEFINE_string('gpu_type', 't4',
                                'Which GPU type to use, e.g., t4, v100, a100.')
_NUM_CPU = flags.DEFINE_integer('num_cpu', 4, 'The number of CPUs.')
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')
_UPDATE = flags.DEFINE_string('update', 'auto',
                              'argument for mdrun -update flag; choices are "auto", "cpu", "gpu".')
_RAM_GB = flags.DEFINE_integer('ram_gb', 4, 'Amount of RAM requested in GiB.')
_MAX_RESTARTS = flags.DEFINE_integer('max_restarts', 2,
                                     'Maximum number of restarts we will attempt if a simulation fails.')
_LOG_LEVEL = flags.DEFINE_string('log_level', 'INFO', 'Set level for logging '
                                 '(DEBUG, INFO, ERROR). Default is "INFO".')


def _gromacs_pkgs() -> xm.Packageable:
  """Returns a contructed python container for running Gromacs simulations.
  """
  # Experiment expects output dir in '/gcs/...' format.
  executable_args = {
      'top_path': _TOP_PATH.value,
      'mdp_path': _MDP_PATH.value,
      'out_path': _OUT_PATH.value,
      'apo_dir': _APO_DIR.value,
      'equil_time': _EQUIL_TIME.value,
      'num_repeats': _NUM_REPEATS.value,
      'target_precision_wat': _TARGET_PRECISION_WAT.value,
      'target_precision_pro': _TARGET_PRECISION_PRO.value,
      'freq_precision_eval': _FREQ_PRECISION_EVAL.value,
      'patience': _PATIENCE.value,
      'min_num_transitions': _MIN_NUM_TRANSITIONS.value,
      'num_mpi': _NUM_MPI.value,
      'num_threads': _NUM_THREADS.value,
      'update': _UPDATE.value,
      'max_restarts': _MAX_RESTARTS.value,
      'log_level': _LOG_LEVEL.value,
  }
  return [
      xm.python_container(
          executor_spec=xm_local.Vertex.Spec(),
          args=executable_args,
          base_image=_BASE_IMAGE.value,
          entrypoint=xm.ModuleName('abfe_base_protocol.main'),
          use_deep_module=True,
      )
  ]


def _parse_lig_ids(lig_ids_str: str) -> List:
  if '-' in lig_ids_str:  # Assume range.
    start_id, end_id = lig_ids_str.split('-')
    lig_ids = range(int(start_id), int(end_id)+1, 1)
    if sum(1 for _ in lig_ids) < 1:
      raise ValueError(f'--lig_ids input "{lig_ids_str}" does not specify a range correctly')
  elif ',' in lig_ids_str:  # Assume comma-separated list of str.
    lig_ids = [i for i in lig_ids_str.split(',')]
  else:  # Assume single digit as str.
    lig_ids = [lig_ids_str]
  return lig_ids


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  flags.mark_flag_as_required('base_image')
  flags.mark_flag_as_required('top_path')
  flags.mark_flag_as_required('mdp_path')
  flags.mark_flag_as_required('out_path')
  flags.mark_flag_as_required('lig_ids')
  
  # Perform some input validation.
  if _NUM_CPU.value != _NUM_THREADS.value * _NUM_MPI.value:
    logging.error(
        'Number of total threads does NOT equal to the number of CPUs. '
        'Performance might suffer.'
    )

  # Get name of all the ligand{id} folders with input structures and
  # topologies for protein-ligand complexes.
  lig_ids = _parse_lig_ids(_LIG_IDS.value)
  lig_dirs = [f'ligand_{i}' for i in lig_ids]

  with xm_local.create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    # Package the Docker container with Gromacs and PMX installed.
    [executable] = experiment.package(_gromacs_pkgs())
    # Define resources required (per work unit).
    job_requirements = xm.JobRequirements({xm.GpuType[_GPU_TYPE.value]: 1},
                                          cpu=_NUM_CPU.value,
                                          ram=_RAM_GB.value * xm.GiB)
    # Instantiate GCP executor.
    executor = xm_local.Vertex(requirements=job_requirements)
    
    # Create a separate work unit for each ligand (i.e. affinity calculation).
    # Each calculation will be executed independently.
    for lig_dir in lig_dirs:
      experiment.add(xm.Job(executable, executor, args={"lig_dir": lig_dir}))


if __name__ == '__main__':
  app.run(main)
