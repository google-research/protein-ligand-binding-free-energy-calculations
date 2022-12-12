r"""The main program for running gromacs ABFE calculations with
a non-equilibrium protocol using .
"""

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
_N_REPEATS = flags.DEFINE_integer('n_repeats', 5,
                                'The number of simulation repeats.')
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')
_LOG_LEVEL = flags.DEFINE_string('log_level', 'INFO', 'Set level for logging '
                                 '(DEBUG, INFO, ERROR). Default is "INFO".')


def mdrun(tpr, pmegpu=True, transition=False):
  """Wrapper for gmxapi.mdrun with predefined scenarios optimized for different 
  types of simulations.

  Args:
    tpr (str): path to TPR file.
    pmegpu (bool): whether to run PME calculations on the GPU. Default is True.
      Note this is not possible with certain integrators.
    transition (bool): whether we are running a non-equilibrium transition. 
      Default is False.

  Returns:
    object: StandardOperationHandle returned by gmxapi.mdrun.
  """

  # Get gmx executable.
  gmxexec = gmxapi.commandline.cli_executable().as_posix()

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

    subprocess.call([f'{gmxexec}', 'mdrun',
                     '-s', f'{tpr}',
                     '-x', f'{path}/traj_{n}.xtc',
                     '-o', f'{path}/traj_{n}.trr',
                     '-c', f'{path}/confout_{n}.gro',
                     '-e', f'{path}/ener_{n}.edr',
                     '-g', f'{path}/md_{n}.log',
                     '-cpo', f'{path}/state_{n}.cpt',
                     '-dhdl', f'{path}/dhdl_{n}.xvg',
                     '-nb', 'gpu',
                     '-pme', _pme,
                     '-pmefft', _pmefft,
                     '-bonded', _bonded,
                     '-ntmpi', str(_NUM_MPI.value),
                     '-ntomp', str(_NUM_THREADS.value),
                     '-pin', 'on'],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.STDOUT)
  else:
    subprocess.call([f'{gmxexec}', 'mdrun',
                     '-s', f'{tpr}',
                     '-x', f'{path}/traj.xtc',
                     '-o', f'{path}/traj.trr',
                     '-c', f'{path}/confout.gro',
                     '-e', f'{path}/ener.edr',
                     '-g', f'{path}/md.log',
                     '-cpo', f'{path}/state.cpt',
                     '-dhdl', f'{path}/dhdl.xvg',
                     '-nb', 'gpu',
                     '-pme', _pme,
                     '-pmefft', _pmefft,
                     '-bonded', _bonded,
                     '-ntmpi', str(_NUM_MPI.value),
                     '-ntomp', str(_NUM_THREADS.value),
                     '-pin', 'on'],
                     stdout=subprocess.DEVNULL,
                     stderr=subprocess.STDOUT)

  # If transition, clean up files we won't need.
  if transition:
    for f in [f'{path}/traj_{n}.xtc', 
              f'{path}/traj_{n}.trr', 
              f'{path}/confout_{n}.gro', 
              f'{path}/ener_{n}.edr', 
              f'{path}/state_{n}.cpt']:
      if os.path.isfile(f):
        os.remove(f)


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


def run_tpr(tpr_file, pmegpu, transition, max_restarts):
  logging.info(f'  --> {tpr_file}')
  start_time = time.time()
  
  # Check if completed already.
  done = mdrun_completed(tpr_file=tpr_file, transition=transition)
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
    mdrun(tpr_file, pmegpu=pmegpu, transition=transition)
    # Check if simulation completed successfully.
    done = mdrun_completed(tpr_file=tpr_file, transition=transition)
    
  end_time = time.time()
  time_elapsed = parse_time(start_time, end_time)
  if done:
    logging.info(f"  ... simulation completed successfully in {time_elapsed} with {num_restarts-1} restarts")
  else:
    logging.info(f"  !!! max restarts ({max_restarts}) reached, mdrun did not complete successfully")


def run_all_tprs(tpr_files, pmegpu=True, transition=False):
  for tpr_file in tpr_files:
    run_tpr(tpr_file, pmegpu, transition, max_restarts=2)

  
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
  fe.equilTime = 2000.
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
  run_all_tprs(tpr_files, pmegpu=False, transition=False)
  
  # Short equilibrations.
  logging.info('Running equilibration.')
  tpr_files = fe.prepare_simulation(simType='eq_posre', prevSim='em')
  run_all_tprs(tpr_files, pmegpu=True, transition=False)
  
  # Equilibrium simulations.
  logging.info('Running production equilibrium simulations.')
  tpr_files = fe.prepare_simulation(simType='eq', prevSim='eq_posre')
  run_all_tprs(tpr_files, pmegpu=True, transition=False)

  # Non-equilibrium simulations.
  logging.info('Running alchemical transitions.')
  tpr_files = fe.prepare_simulation(simType='transitions')
  run_all_tprs(tpr_files, pmegpu=True, transition=True)
  
  # Analysis.
  logging.info('Analyzing results.')
  fe.run_analysis(ligs=[_LIG_DIR.value])
  fe.analysis_summary(ligs=[_LIG_DIR.value])
  # Write results to file.
  fe.resultsSummary.to_csv(f"{_OUT_PATH.value}/results.csv", index=False)
  # Write breakdown of all terms contributing to overall dG.
  fe.resultsAll.to_csv(f"{_OUT_PATH.value}/dg_terms.csv", index=True)

  logging.info('Job completed.')


if __name__ == '__main__':
  app.run(main)
  
