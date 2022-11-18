r"""The main program for running gromacs simulation with a TPR input.

This module is simply for testing running gromacs with XCloud and it is subject
to modification.
"""

import os
import time

from absl import app
from absl import flags
from absl import logging

import gmxapi

import numpy as np
from pmx.AbsoluteDG import AbsoluteDG

_MDP_PATH = flags.DEFINE_string('mdp_path', None, 'Path to MDP files.')
_TOP_PATH = flags.DEFINE_string('top_path', None, 'Path to GRO/TOP files.')
_OUT_PATH = flags.DEFINE_string('out_path', None, 'Output directory.')
_LIG_DIR = flags.DEFINE_string('lig_dir', None, 'Name of directory with ligand TOP/GRO files.')
_PROT_DIR = flags.DEFINE_string('prot_dir', None, 'Name of directory with protein TOP/GRO files.')
_N_REPEATS = flags.DEFINE_integer('n_repeats', 5,
                                'The number of simulation repeats.')
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')

# Sets a session ID for this session.
_SESSION_ID = str(int(time.time()))


# Monkey patch this session ID into the gmxapi.operation.ResourceManager.
# Hack. Do not use in the user code.
# This is temporarily added to allow users to start from a new set of working
# directories for all the mdrun launches. GMX API doesn't expose the
# underlying resource manager or context yet.
def operation_id(self):
  # pylint: disable=protected-access
  return f'{self._base_operation_id}_{_SESSION_ID}'


gmxapi.operation.ResourceManager.operation_id = property(operation_id)


def mdrun(tpr, pmegpu=True, nsteps=None, transition=False):
  """Wrapper for gmxapi.mdrun with predefined scenarios optimized for different 
  types of simulations.

  Args:
    tpr (str): path to TPR file.
    pmegpu (bool): whether to run PME calculations on the GPU. Default is True.
      Note this is not possible with certain integrators.
    nsteps (int): number of integration steps to run, overwrites nsteps defined
      by the MDP file used to generate the TPR. Default is None (do not modify 
      TPR).
    transition (bool): whether we are running a non-equilibrium transition. 
      Default is False.

  Returns:
    object: StandardOperationHandle returned by gmxapi.mdrun.
  """
  # Load TPR file.
  input_tpr = gmxapi.read_tpr(tpr)
  if nsteps is not None:
    input_tpr = gmxapi.modify_input(input=input_tpr, 
                                    parameters={'nsteps': nsteps})    

  # Get path to TPR.
  path = "/".join(tpr.split('/')[:-1])
  
  # Specify command line flags and arguments to be passed to mdrun.
  # Note: energy minimization cannot run PME on GPU.
  if pmegpu:
    _pme = 'gpu'
    _pmefft = 'gpu'
    _bonded = 'gpu'
  else:
    _pme = 'auto'
    _pmefft = 'auto'
    _bonded = 'auto'

  # If we are running a non-equilibrium transition, we want to get the dhdl
  # file and keep only that. We then use the presence of dhdl_{n}.xvg to 
  # determine if the simulation has been ran already.
  if transition:
    n = int(tpr.split('/')[-1].split('.')[0].replace('ti', ''))
    md = gmxapi.mdrun(input_tpr,
                      runtime_args={'-nb': 'gpu', 
                                    '-pme': _pme, 
                                    '-pmefft': _pmefft, 
                                    '-bonded': _bonded,
                                    '-ntmpi': str(_NUM_MPI.value),
                                    '-ntomp': str(_NUM_THREADS.value),
                                    '-pin': 'on',
                                    '-x': f'{path}/traj_{n}.xtc',
                                    '-o': f'{path}/traj_{n}.trr',
                                    '-c': f'{path}/confout_{n}.gro',
                                    '-e': f'{path}/ener_{n}.edr', 
                                    '-g': f'{path}/md_{n}.log',
                                    '-cpo': f'{path}/state_{n}.cpt',
                                    '-dhdl': f'{path}/dhdl_{n}.xvg'
                                    }
                      )
  else:
    md = gmxapi.mdrun(input_tpr,
                      runtime_args={'-nb': 'gpu', 
                                    '-pme': _pme, 
                                    '-pmefft': _pmefft, 
                                    '-bonded': _bonded,
                                    '-ntmpi': str(_NUM_MPI.value),
                                    '-ntomp': str(_NUM_THREADS.value),
                                    '-pin': 'on',
                                    '-x': f'{path}/traj.xtc',
                                    '-o': f'{path}/traj.trr',
                                    '-c': f'{path}/confout.gro',
                                    '-e': f'{path}/ener.edr', 
                                    '-g': f'{path}/md.log',
                                    '-cpo': f'{path}/state.cpt'
                                    }
                      )

  # Run the simulation.
  md.run()

  # If transition, clean up files we won't need.
  if transition:
    for f in [f'{path}/traj_{n}.xtc', 
              f'{path}/traj_{n}.trr', 
              f'{path}/confout_{n}.gro', 
              f'{path}/ener_{n}.edr', 
              f'{path}/md_{n}.log',  
              f'{path}/state_{n}.cpt']:
      if os.path.isfile(f):
        os.remove(f)

  return md


def mdrun_completed(tpr_file: str, transition: bool = False) -> bool:
  """Checks whether energy minimization completed successfully.
  
  Args:
    tpr (str): path to TPR file.
    transition (bool): whether we are checking completion for a non-equilibrium 
      transition. Default is False.

  Returns:
    bool: whether the TPR has been run successfully.
  """
  # If we're chekcing the completion of a non-eq transition, we check that (i)
  # the right dhdl.xvg file exists, and that (ii) it contains info up to the 
  # end of the transition. Otherwise, we assume it crashed and needs to be
  # completed.
  if transition:
    # Get the index of the transition.
    n = int(tpr_file.split('/')[-1].split('.')[0].replace('ti', ''))
    # Get dhdl file path.
    dhdl = "/".join(tpr_file.split("/")[:-1]) + f"/dhdl_{n}.xvg"
    # If dhdl file exists, check it's complete.
    if os.path.isfile(dhdl):
      input_tpr = gmxapi.read_tpr(tpr_file)
      nsteps = input_tpr.output.parameters.result()['nsteps']
      dt = input_tpr.output.parameters.result()['dt']
      expected_final_time = nsteps * dt
      
      with open(dhdl, 'r') as f:
        lines = f.readlines()
      
      try:
        actual_final_time = float(lines[-1].split()[0])
        if expected_final_time - actual_final_time < 1e-6:
          return True
      except:
        return False
      
    return False
  else:
    # (maldeghi): IIRC Gromacs would output the GRO file only for mdruns that
    # did not crash/errored. We can make this stricted by checking the log
    # file too.
    gro = "/".join(tpr_file.split("/")[:-1]) + "/confout.gro"
    if os.path.isfile(gro):
      return True
    else:
      return False


def run_all_tprs(tpr_files, pmegpu=True, transition=False):
  for tpr_file in tpr_files:
    # If minimization has been run already, skip.
    if mdrun_completed(tpr_file=tpr_file, transition=transition):
      print(f"`{tpr_file}` already ran successfully")
      continue
    # Run the simulation (with reduced number of steps if needed)
    _ = mdrun(tpr_file, pmegpu=pmegpu, nsteps=None)


def setup_abfe_obj_and_output_dir():

  # Initialize the free energy environment object. It will store the main 
  # parameters for the calculations.
  fe = AbsoluteDG(mdpPath=_MDP_PATH.value,  # Path to the MDP files.
                  structTopPath=_TOP_PATH.value,  # Path to structures and topologies.
                  ligList=[_LIG_DIR.value],  # Folders with protein-ligand input files.
                  apoCase=_PROT_DIR.value,  # Folders with protein files.
                  bDSSB=False,
                  gmxexec=gmxapi.commandline.cli_executable().as_posix())

  # Set the workpath in which simulation input files will be created.
  fe.workPath = _OUT_PATH.value
  # Set the number of replicas (i.e., number of equilibrium simulations per state).
  fe.replicas = _N_REPEATS.value

  # Prepare the directory structure with all simulations steps required.
  fe.simTypes = ['em',  # Energy minimization.
                'eq_posre',  # Equilibrium sim with position restraints.
                'eq',  # Equilibrium simulation.
                'transitions']  # Alchemical, non-equilibrium simulations.

  # ps to discard from equilibrium simulations when preparing the input for 
  # the non-equilibrium ones.
  fe.equilTime = 2000.
  # Whether to generate TPRs from extracted frames and rm GRO files.
  fe.bGenTiTpr = True

  fe.prepareFreeEnergyDir()
  return fe


def main(_):
  logging.info('Job started.')
  
  flags.mark_flag_as_required('mdp_path')
  flags.mark_flag_as_required('top_path')
  flags.mark_flag_as_required('out_path')
  flags.mark_flag_as_required('lig_dir')

  logging.info('Instantiate AbsoluteDG object.')
  fe = setup_abfe_obj_and_output_dir()

  logging.info('Change workdir to %s', _OUT_PATH.value)
  os.chdir(_OUT_PATH.value)

  logging.info('Assemble simulation systems.')
  # Assemble the systems: build Gromacs structure and topology for the 
  # ligand+water and ligand+protein+water systems.
  fe.assemble_systems()

  # Define the simulation boxes, fill them with water molecules, and add ions to 
  # neutralize the system and reach desired NaCl concentration (0.15 M by default).
  fe.boxWaterIons()

  # Energy minimization.
  logging.info('Run energy minimization.')
  tpr_files = fe.prepare_simulation(simType='em')
  # Read the TPR files and run all minimizations.
  run_all_tprs(tpr_files, pmegpu=False, transition=False)
  
  # Short equilibrations.
  logging.info('Run equilibration.')
  tpr_files = fe.prepare_simulation(simType='eq_posre', prevSim='em')
  run_all_tprs(tpr_files, pmegpu=True, transition=False)
  
  # Equilibrium simulations.
  logging.info('Run production equilibrium simulations.')
  tpr_files = fe.prepare_simulation(simType='eq', prevSim='eq_posre')
  run_all_tprs(tpr_files, pmegpu=True, transition=False)

  # Non-equilibrium simulations.
  logging.info('Run alchemical transitions.')
  tpr_files = fe.prepare_simulation(simType='transitions')
  run_all_tprs(tpr_files, pmegpu=True, transition=True)
  
  # Analysis.
  logging.info('Analyze results.')
  fe.run_analysis(ligs=[_LIG_DIR.value])
  df_results = fe.resultsSummary
  df_results.to_csv(f"{_OUT_PATH.value}/results.csv", index=False)

  logging.info('Job completed.')


if __name__ == '__main__':
  app.run(main)
  
