r"""The main program for running gromacs ABFE calculations with
a non-equilibrium protocol and Boresch restraints.
"""

from typing import List
import os
import subprocess
import time

from absl import app
from absl import flags
from absl import logging

import gmxapi
from pmx.AbsoluteDG import AbsoluteDG


_MDP_PATH = flags.DEFINE_string('mdp_path', None, 'Path to MDP files.')
_TOP_PATH = flags.DEFINE_string('top_path', None, 'Path to GRO/TOP files.')
_OUT_PATH = flags.DEFINE_string('out_path', None, 'Output directory.')
_LIG_DIR = flags.DEFINE_string('lig_dir', None, 'Name of directory with ligand TOP/GRO files.')
_APO_DIR = flags.DEFINE_string('apo_dir', None, 'Name of directory with TOP/GRO files for apo protein.')
_EQUIL_TIME = flags.DEFINE_float('equil_time', 1000.1, 'Equilibration time (in ps) to discard from '
                                 'the equilibrium simulations.')
_N_REPEATS = flags.DEFINE_integer('n_repeats', 5,
                                'The number of simulation repeats.')
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')
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
  try:
    process = subprocess.run(parameter_pack, check=True, capture_output=True)
  except subprocess.CalledProcessError as e:
    # Mark simulation as failed.
    open(os.path.join(simpath, '_FAILED'), 'w+').close()
    # Log the failure.
    logging.error("  ... simulation failed with exception: %s", str(e))
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


def run_tpr(tpr_file, pmegpu, max_restarts):
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
    run_tpr(tpr_file, pmegpu, max_restarts=2)

  
def parse_time(start, end):
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


def setup_abfe_obj_and_output_dir():

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
  fe.replicas = _N_REPEATS.value

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

  # Energy minimization.
  logging.info('Running energy minimizations.')
  tpr_files = fe.prepare_simulation(simType='em')
  # Read the TPR files and run all minimizations.
  run_all_tprs(tpr_files, pmegpu=False)
  
  # Short equilibrations.
  logging.info('Running equilibration.')
  tpr_files = fe.prepare_simulation(simType='eq_posre', prevSim='em')
  run_all_tprs(tpr_files, pmegpu=True)
  
  # Equilibrium simulations.
  logging.info('Running production equilibrium simulations.')
  tpr_files = fe.prepare_simulation(simType='eq', prevSim='eq_posre')
  run_all_tprs(tpr_files, pmegpu=True)

  # Non-equilibrium simulations.
  logging.info('Running alchemical transitions.')
  tpr_files = fe.prepare_simulation(simType='transitions')
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
  
