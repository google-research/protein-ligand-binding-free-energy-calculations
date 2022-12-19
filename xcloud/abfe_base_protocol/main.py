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

r"""The main program for running gromacs ABFE calculations with
a non-equilibrium protocol and Boresch restraints.
"""

from typing import List, Dict, Tuple
import os
import subprocess
import time
import random
import glob
import scipy

from absl import app
from absl import flags
from absl import logging

import gmxapi
import pmx
from pmx.AbsoluteDG import AbsoluteDG


# Flags about paths and input folders.
_TOP_PATH = flags.DEFINE_string('top_path', None, 'Path to folder with GRO/TOP files. We expect this directory will have '
                                'one subfolder for each calculation, with the inputs for different ligands, and '
                                'optionally for the apo protein simulations.')
_MDP_PATH = flags.DEFINE_string('mdp_path', None, 'Path to MDP files.')
_OUT_PATH = flags.DEFINE_string('out_path', None, 'Output directory.')
_LIG_DIR = flags.DEFINE_string('lig_dir', None, 'Name of directory with ligand TOP/GRO files.')
_APO_DIR = flags.DEFINE_string('apo_dir', None, 'Name of directory with TOP/GRO files for apo protein.')
# Flags about simulation parameters.
_EQUIL_TIME = flags.DEFINE_float('equil_time', 1000.1, 'Equilibration time (in ps) to discard from '
                                 'the equilibrium simulations.')
_NUM_REPEATS = flags.DEFINE_integer('num_repeats', 5,
                                'The number of simulation repeats.')
# Flags about early stopping.
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
# Flags about paralellization.
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')
# Flags about other stuff.
_MAX_RESTARTS = flags.DEFINE_integer('max_restarts', 2,
                                     'Maximum number of restarts we will attempt if a simulation fails.')
_LOG_LEVEL = flags.DEFINE_string('log_level', 'INFO', 'Set level for logging '
                                 '(DEBUG, INFO, ERROR). Default is "INFO".')


def _build_mdrun_args(tpr: str, pmegpu: bool) -> List:
  """Builds the gmx mdrun command to be executed.

  Args:
    tpr (str): path to TPR file.
    pmegpu (bool): whether to run PME calculations on the GPU. Default is True.
      Note this is not possible with certain integrators.

  Returns:
    List: command line to be executed with subprocess.
  """

  # Get gmx executable.
  gmxexec = gmxapi.commandline.cli_executable().as_posix()

  # Get path to TPR.
  simpath = os.path.dirname(tpr)

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

  parameter_pack = [f'{gmxexec}', 'mdrun',
                   '-s', f'{tpr}',
                   '-x', f'{simpath}/traj.xtc',
                   '-o', f'{simpath}/traj.trr',
                   '-c', f'{simpath}/confout.gro',
                   '-e', f'{simpath}/ener.edr',
                   '-g', f'{simpath}/md.log',
                   '-cpo', f'{simpath}/state.cpt',
                   '-dhdl', f'{simpath}/dhdl.xvg',
                   '-nb', 'gpu',
                   '-pme', _pme,
                   '-pmefft', _pmefft,
                   '-bonded', _bonded,
                   '-ntmpi', str(_NUM_MPI.value),
                   '-ntomp', str(_NUM_THREADS.value),
                   '-pin', 'on']

  return parameter_pack


def mdrun(tpr: str, pmegpu: bool = True) -> None:
  """Wrapper mdrun with predefined scenarios optimized for different 
  types of simulations.

  Args:
    tpr (str): path to TPR file.
    pmegpu (bool): whether to run PME calculations on the GPU. Default is True.
      Note this is not possible with certain integrators.

  Returns:
    None
  """

  # Get path to TPR.
  simpath = os.path.dirname(tpr)

  # Get mdrun arguments.
  parameter_pack = _build_mdrun_args(tpr, pmegpu)

  # Run mdrun.
  process = subprocess.run(parameter_pack, check=False, capture_output=True, text=True)
  if process.returncode != 0:
    # Mark simulation as failed.
    open(os.path.join(simpath, '_FAILED'), 'w+').close()
    # Log the failure.
    logging.error(f'  ... simulation failed (returned non-zero exit status {process.returncode})')
    # Show the last output from Gromacs for ease of debugging.
    logging.error('='*30 + 'last 20 lines of Gromacs stderr' + '='*30)
    _gmx_stderr = "\n".join(process.stderr.splitlines()[-20:])
    logging.error(_gmx_stderr)
    logging.error('='*30 + '='*31 + '='*30)
    return

  # Mark simulation as completed.
  open(os.path.join(simpath, '_COMPLETED'), 'w+').close()
  # Rm previous _FAILED, as it means we eventually completed the simulation.
  if os.path.isfile(os.path.join(simpath, '_FAILED')):
    os.remove(os.path.join(simpath, '_FAILED'))

  # Log gromacs output for debugging purposes. Not logging to INFO becuase
  # it's too much output. Also note that gromacs prints normal info
  # messages to stderr.
  if not process.stdout:
    logging.debug("Completed simulation (stdout): \n %s", process.stdout)
  if not process.stderr:
    logging.debug("Completed simulation (stderr): \n %s", process.stderr)


def _mdrun_completed(simpath):
  if os.path.isfile(os.path.join(simpath, '_COMPLETED')):
    return True
  else:
    return False


def run_tpr(tpr_file: str, pmegpu: bool, max_restarts: int) -> None:
  """Run the simulation associated with a TPR file. But it checks if the simulation
  was already run, and tries to restart it if it crashes.

  Args:
      tpr_file (str): absolute path to TPR file.
      pmegpu (bool): whether to use PME on GPU.
      max_restarts (int): max number of restart attempts for crashing simulations.
  """
  logging.info(f'  --> {tpr_file}')
  start_time = time.time()
  
  # Check if completed already. This check is meant to be useful if
  # the job was unexpectedly killed while mdrun was being executed. 
  done = _mdrun_completed(os.path.dirname(tpr_file))
  if done:
    logging.info(f"  ... mdrun already completed successfully, skipping")
    return

  # If run not previously completed (i.e. not a job restart)
  # then execute mdrun while allowing for a few re-starts in case
  # of crashes.
  num_restarts = 0
  while not done and num_restarts <= max_restarts:
    # Run the simulation. If previous attempt failed
    logging.info(f'  ... executing mdrun')
    num_restarts += 1
    mdrun(tpr_file, pmegpu=pmegpu)
    # Check if simulation completed successfully.
    done = _mdrun_completed(os.path.dirname(tpr_file))
    
  end_time = time.time()
  time_elapsed = parse_time(start_time, end_time)
  if done:
    logging.info(f"  ... simulation completed successfully in {time_elapsed} with {num_restarts-1} restarts")
  else:
    logging.info(f"  !!! max restarts ({max_restarts}) reached, mdrun did not complete successfully")


def run_all_tprs(tpr_files, pmegpu=True):
  for tpr_file in tpr_files:
    run_tpr(tpr_file, pmegpu, max_restarts=_MAX_RESTARTS.value)


def run_all_transition_tprs_with_early_stopping(tpr_files: List, pmegpu: bool = True, patience: int = 2):
  """Run non-equilibrium simulations with en early stopping mechanism. If we desired 
  precision (specified by _TARGET_PRECISION) is reached in the N consecutive iterations 
  (as specified by `patience`), we terminate the execution of additional simulations. 
  The interval at which we check the precision is specified by _FREQ_PRECISION_EVAL.

  Args:
      tpr_files (List): list of TPR files with absolute paths.
      pmegpu (bool, optional): whether to run PME on the GPU. Defaults to True.
      patience (int, optional): number of consecutive iterations in which target 
        precision is reached that are allowed before terminating further mdrun 
        executions. Defaults to 2.
  """

  # Split list of tpr files in the water/protein, state A/B, run N.
  # Also, shuffle tpr lists. This is done to ensure uniform sampling of 
  # starting points.
  tpr_tree = _build_tpr_tree(tpr_files, shuffle=True)

  def _execute_sims_with_early_stopping_loop(env: str, target_precision: float, patience: int) -> None:
    """Runs non-equil transitions with earaly stopping for either the water or 
    protein environment.

    Args:
        env (str): ligand environment ('water' or 'protein').
        target_precision (float): desired precision in kJ/mol.
        patience (int): patience counter.
    """
    # Validate args.
    _valid_envs = ['water', 'protein']
    if env not in _valid_envs:
        raise ValueError('`env` can only be "water" or "protein"')

    def _run_multiple_transitions(num_transitions: int) -> int:
      """Runs the specified number of transitions.

      Args:
          num_transitions (int): number of transitions to run.

      Returns:
          int: actual number of transitions run.
      """
      _num_transitions_actually_run = 0
      # Run N non-equil sims for both states A and B, and all repeats.
      for _ in range(num_transitions):
        for state in tpr_tree[env].keys():
          for run in tpr_tree[env][state].keys():
            # Check the list is not empty (i.e. we ran all sims already).
            if tpr_tree[env][state][run]:
              tpr_file = tpr_tree[env][state][run].pop(0)
              run_tpr(tpr_file, pmegpu, max_restarts=_MAX_RESTARTS.value)
              _num_transitions_actually_run += 1
      return _num_transitions_actually_run

    def _evaluate_precision_and_update_patience(target_precision: float, _patience: int) -> int:
      # Run analysis and get precision.
      logging.info(f"Evaluating precision for {env} dG")
      precision = _estimate_dg_precision(env=env)
      
      # If target precision reached, decrease patience counter.
      if precision < target_precision:
        _patience -= 1
        logging.info(f"  Target precision reached ({precision:.2f} < {target_precision:.2f} kJ/mol)")
        logging.info(f"  Patience left: {_patience}")
      else:
        # Reset patience. This ensures that only if precistion < target_precision in consecutive
        # iterations we stop early. If we hit the target, then miss it with more data, we reset
        # the patience counter.
        _patience = patience
        logging.info(f"  Precision ({precision:.2f} kJ/mol) above target one ({target_precision:.2f} kJ/mol)")
      
      return _patience

    # Running patience value.
    _patience = patience

    # First, run the minimum number of transitions we want to guarantee are run.
    if _MIN_NUM_TRANSITIONS.value > 0:
      _ = _run_multiple_transitions(_MIN_NUM_TRANSITIONS.value)
      # Run analysis and get precision.
      _patience = _evaluate_precision_and_update_patience(target_precision, _patience)

    # Then, run the other transitions, _FREQ_PRECISION_EVAL at a time, until
    # we achieve desired precision for N iterations, as defined by the
    # patience param.
    while _patience > 0:
      # Run N non-equil sims for both states A and B, and all repeats.
      _num_transitions_actually_run = _run_multiple_transitions(_FREQ_PRECISION_EVAL.value)
      # Safety mechanism to exit the loop. E.g., we ran all simulations
      # such that there is no tpr left to run. 
      if _num_transitions_actually_run < 1:
        logging.info(f"  Available non-equil simulations in {env} environment exhausted")
        return
      
      # Run analysis and get precision.
      _patience = _evaluate_precision_and_update_patience(target_precision, _patience)
      
    logging.info(f"  Terminating execution of non-equilibrium transitions in {env} environment")

  # Run both water and protein sims until precision reached or run out of transitions.
  _execute_sims_with_early_stopping_loop(env='water', 
                                         target_precision=_TARGET_PRECISION_WAT.value, 
                                         patience=patience)
  _execute_sims_with_early_stopping_loop(env='protein', 
                                         target_precision=_TARGET_PRECISION_PRO.value, 
                                         patience=patience)


def _estimate_dg_precision(env: str = 'water', T: float = 298.15) -> float:
  """Estimate the standard error of the dG estimate across independent repeats.

  Args:
      env (str, optional): the ligand environment, 'water' or 'protein'. Defaults to 'water'.
      T (float, optional): the temperature in Kelvin. Defaults to 298.15.

  Returns:
      float: standard error of the mean for dG estimates across separate repeats.
  """

  dg_estimates = []  # List used to store dG estimates from different repeats.

  # Estimate dG for each repeat.
  for i in range(1, _NUM_REPEATS.value + 1):
    # Get all available data from gromacs.
    filesA = glob.glob(f'{_OUT_PATH.value}/{_LIG_DIR.value}/{env}/stateA/run{i}/transitions/frame*/dhdl.xvg')
    filesB = glob.glob(f'{_OUT_PATH.value}/{_LIG_DIR.value}/{env}/stateB/run{i}/transitions/frame*/dhdl.xvg')
    # Read files and compute integrated work values.
    workA = pmx.analysis.read_dgdl_files(filesA, lambda0=0, invert_values=False, verbose=False, sigmoid=0.0)
    workB = pmx.analysis.read_dgdl_files(filesB, lambda0=1, invert_values=False, verbose=False, sigmoid=0.0)
    # Estimate dG value with BAR.
    bar = pmx.estimators.BAR(workA, workB, T=T, nboots=0, nblocks=1)
    # Append estimate.
    dg_estimates.append(bar.dg)
  
  # Returns standard error across repeats.
  return scipy.stats.sem(dg_estimates)


def _build_tpr_tree(tpr_list: List, shuffle: bool = True) -> Dict:
  """Takes a list of paths to TPR files and splits them by environment (water vs protein),
  state (A vs B), and repeat number. The resulting lists are put into a dict of dicts of lists
  in tree-like structure.

  Args:
      tpr_list (List): list of TPR files with abolute paths.
      shuffle (bool, optional): whether to shuffle the resulting lists. We do this to ensure 
        uniform sampling of starting points when running the simulations one by one. Defaults to True.

  Returns:
      Dict: dictionary with tree-like structure (dict[env][state][repeat], 
        e.g. dict['water']['stateA']['run3']), containing lists of TPR files organized by
        environment, state, and repeat number.
  """
  tpr_dict = {}
  for env in ['water', 'protein']:
    tpr_dict[env] = {}
    for state in ['stateA', 'stateB']:
      tpr_dict[env][state] = {}
      for run in [f'run{n}' for n in range(1, _NUM_REPEATS.value + 1)]:
        tpr_dict[env][state][run] = [tpr for tpr in tpr_list if f'{env}/{state}/{run}' in tpr]
        if shuffle:
          random.shuffle(tpr_dict[env][state][run])
  return tpr_dict


def parse_time(start: float, end: float) -> str:
  """Makes a readable string for elapsed time.

  Args:
      start (float): start time from time.time().
      end (float): end time from time.time().

  Returns:
      str: elapsed time described in hours, minutes, seconds.
  """
  elapsed = end - start  # elapsed time in seconds
  if elapsed <= 1.0:
    ms = elapsed * 1000.
    time_string = f"{ms:.1f} ms"
  elif 1.0 < elapsed <= 60.0:
    time_string = f"{elapsed:.1f} s"
  elif 60.0 < elapsed <= 3600.0:
    m, s = divmod(elapsed, 60)
    time_string = f"{m:.0f} min, {s:.0f} s"
  else:
    h, m = divmod(elapsed, 3600)
    m, s = divmod(m, 60)
    time_string = f"{h:.0f} h, {m:.0f} min, {s:.0f} s"
  return time_string


def setup_abfe_obj_and_output_dir() -> AbsoluteDG:
  """Instantiates PMX object handling the setup and analysis of the calculation.

  Returns:
      AbsoluteDG: instance handling the calculation setup/analysis.
  """

  # Initialize the free energy environment object. It will store the main 
  # parameters for the calculations.
  fe = AbsoluteDG(mdpPath=_MDP_PATH.value,  # Path to the MDP files.
                  structTopPath=_TOP_PATH.value,  # Path to structures and topologies.
                  ligList=[_LIG_DIR.value],  # Folders with protein-ligand input files.
                  apoCase=_APO_DIR.value,  # Folders with apo protein files.
                  bDSSB=False,
                  gmxexec=gmxapi.commandline.cli_executable().as_posix())

  # Set the workpath in which simulation input files will be created.
  fe.workPath = _OUT_PATH.value
  # Set the number of replicas (i.e., number of equilibrium simulations per state).
  fe.replicas = _NUM_REPEATS.value

  # Prepare the directory structure with all simulations steps required.
  fe.simTypes = ['em',  # Energy minimization.
                 'eq_posre',  # Equilibrium sim with position restraints.
                 'eq',  # Equilibrium simulation.
                 'transitions']  # Alchemical, non-equilibrium simulations.

  # Specify the number of equivalent poses available for the ligand due to 
  # simmetry, and which we are unlikely to sample during the simulations due to 
  # the definition of the restraints. 
  # For non-symmetric ligands, this is 1, resulting in no correction. 
  # For e.g., benzene, this can be 2 if rotations around
  # the C6 axis are allowed but ring flips are not. 
  # This choice is not straightforward, but chances are it won't matter much 
  # for our purposes of discriminating binders from non-binders.
  fe.ligSymmetry = 1

  # ps to discard from equilibrium simulations when preparing the input for 
  # the non-equilibrium ones.
  fe.equilTime = _EQUIL_TIME.value
  # Whether to generate TPRs from extracted frames and rm GRO files.
  fe.bGenTiTpr = True

  fe.prepareFreeEnergyDir()
  return fe


def _parse_and_validate_flags() -> List[Tuple]:

  if _NUM_REPEATS.value < 2 and _TARGET_PRECISION_WAT.value > 1e-5:
    logging.warning("! Cannot use early stopping with only 1 repeat. "
                    "Early stopping for water calcs switched off.")
    _TARGET_PRECISION_WAT.value = 0.

  if _NUM_REPEATS.value < 2 and _TARGET_PRECISION_PRO.value > 1e-5:
    logging.warning("! Cannot use early stopping with only 1 repeat. "
                    "Early stopping for water calcs switched off.")
    _TARGET_PRECISION_PRO.value = 0.


def _validate_tpr_generation(sim_stage):
  missing = []
  for env in ['water', 'protein']:
    for state in ['stateA', 'stateB']:
      for n in _NUM_REPEATS.value:
        tpr_file_path = os.path.join(_OUT_PATH.value, _LIG_DIR.value, env, state, f'run{n}', sim_stage, 'tpr.tpr')
        if not os.path.isfile(tpr_file_path):
          missing.append(tpr_file_path)

  if len(missing) > 0:
    logging.error("!!! Not all TPR files were successfully generated !!!")
    for f in missing:
      logging.error(f'Missing "{f}"')
    raise FileExistsError('not all tpr.tpr files were created')
  pass


def _validate_system_assembly():
  missing = []
  for env in ['water', 'protein']:
    for state in ['stateA', 'stateB']:
      ions_file_path = os.path.join(_OUT_PATH.value, _LIG_DIR.value, env, state, 'ions.pdb')
      if not os.path.isfile(ions_file_path):
        missing.append(ions_file_path)
  
  if len(missing) > 0:
    logging.error("!!! Not all systems were successfully assembled !!!")
    for f in missing:
      logging.error(f'Missing "{f}"')
    raise FileExistsError('not all ions.pdb files were created')


def main(_):
  
  flags.mark_flag_as_required('mdp_path')
  flags.mark_flag_as_required('top_path')
  flags.mark_flag_as_required('out_path')
  flags.mark_flag_as_required('lig_dir')

  # Set up logger.
  _log_level = {
      'DEBUG': logging.DEBUG,
      'INFO': logging.INFO,
      'ERROR': logging.ERROR
  }

  # We call this first to create the folder for the job, and place the log
  # file in this path.
  fe = setup_abfe_obj_and_output_dir()

  logging.get_absl_handler().use_absl_log_file('abfe', f'{_OUT_PATH.value}/{_LIG_DIR.value}/')
  flags.FLAGS.mark_as_parsed()
  logging.set_verbosity(_log_level[_LOG_LEVEL.value])
  _parse_and_validate_flags()

  logging.info('===== ABFE calculation started =====')
  logging.info('AbsoluteDG object has been instantiated.')
  logging.info('Changing workdir to %s', _OUT_PATH.value)
  os.chdir(_OUT_PATH.value)

  logging.info('Assembling simulation systems.')
  # Assemble the systems: build Gromacs structure and topology for the 
  # ligand+water and ligand+protein+water systems.
  fe.assemble_systems()

  # Define the simulation boxes, fill them with water molecules, and add ions to 
  # neutralize the system and reach desired NaCl concentration (0.15 M by default).
  logging.info('Building simulation box.')
  fe.boxWaterIons()

  # Check we have the input ions.pdb files for all systems.
  _validate_system_assembly()

  # Energy minimization.
  logging.info('Running energy minimizations.')
  tpr_files = fe.prepare_simulation(simType='em')
  # Check mdrun input has been created.
  _validate_tpr_generation('em')
  # Read the TPR files and run all minimizations.
  run_all_tprs(tpr_files, pmegpu=False)
  
  # Short equilibrations.
  logging.info('Running equilibration.')
  tpr_files = fe.prepare_simulation(simType='eq_posre', prevSim='em')
  _validate_tpr_generation('eq_posre')
  run_all_tprs(tpr_files, pmegpu=True)
  
  # Equilibrium simulations.
  logging.info('Running production equilibrium simulations.')
  tpr_files = fe.prepare_simulation(simType='eq', prevSim='eq_posre')
  _validate_tpr_generation('eq')
  run_all_tprs(tpr_files, pmegpu=True)

  # Non-equilibrium simulations.
  logging.info('Running alchemical transitions.')
  tpr_files = fe.prepare_simulation(simType='transitions')
  if _TARGET_PRECISION_WAT.value > 1e-5 or _TARGET_PRECISION_PRO.value > 1e-5:
    logging.info(f'Using early stopping with target precision of {_TARGET_PRECISION_WAT.value} '
                 f'kJ/mol for water calcs, {_TARGET_PRECISION_PRO.value} kJ/mol for protein calcs, '
                 f'evaluation frequency of {_FREQ_PRECISION_EVAL.value}, and a minimum number of '
                 f'transitions of {_MIN_NUM_TRANSITIONS.value}.')
    run_all_transition_tprs_with_early_stopping(tpr_files, pmegpu=True, patience=_PATIENCE.value)
  else:
    run_all_tprs(tpr_files, pmegpu=True)
  
  # Analysis.
  logging.info('Analyzing results.')
  fe.run_analysis(ligs=[_LIG_DIR.value])
  fe.analysis_summary(ligs=[_LIG_DIR.value])
  # Write results to file.
  fe.resultsSummary.to_csv(os.path.join(_OUT_PATH.value, _LIG_DIR.value, 'results.csv'), index=False)
  # Write breakdown of all terms contributing to overall dG.
  fe.resultsAll.to_csv(os.path.join(_OUT_PATH.value, _LIG_DIR.value, 'dg_terms.csv'), index=True)

  logging.info('Job completed.')


if __name__ == '__main__':
  app.run(main)
  
