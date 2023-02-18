# Copyright 2023 Google LLC
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

r"""Script to summarize the status of ABFE calculations running on GCP."""


from typing import List
import subprocess
import os
import shutil
import tempfile

from absl import app
from absl import flags


# Flags about paths.
_INP_PATH = flags.DEFINE_string(
    'input_path', None, 'GCS path to input files for calculations.'
)
_OUT_PATH = flags.DEFINE_string(
    'output_path', None, 'GCS path to output files of calculations.'
)
_PROTEINS = flags.DEFINE_string(
    'proteins',
    None,
    'Folder names of target proteins (comma-separated list, e.g. "cmet,jnk1").',
)

# Required flag.
flags.mark_flag_as_required('input_path')
flags.mark_flag_as_required('output_path')
flags.mark_flag_as_required('proteins')

_COLORS = {
    'blue': '\033[94m',
    'default': '\033[99m',
    'grey': '\033[90m',
    'yellow': '\033[93m',
    'black': '\033[90m',
    'cyan': '\033[96m',
    'green': '\033[92m',
    'magenta': '\033[95m',
    'white': '\033[97m',
    'red': '\033[91m',
    'endc': '\033[0m',
}


def _parse_protein_str(protein_str: str) -> List:
  return [i for i in protein_str.split(',')]


def argmax(a):
  return max(range(len(a)), key=lambda x: a[x])


def _get_ligand_folders(gs_path):
  process = subprocess.run(
      f'gsutil ls -d {gs_path}', shell=True, capture_output=True, text=True
  )
  ligands = [d.split('/')[-2] for d in process.stdout.split('\n') if d]
  return ligands


def _copy_log_files(gs_path):
  # List all logs.
  process = subprocess.run(
      f'gsutil ls gs://{gs_path}/ligand_*/abfe.*.INFO*',
      shell=True,
      capture_output=True,
      text=True,
  )
  log_files = [f for f in process.stdout.split('\n') if f]

  # If empty just return empty dict.
  if len(log_files) < 1:
    return {}

  # Copy log files.
  temp_dir = tempfile.mkdtemp()
  process = subprocess.run(
      f'gsutil -m cp gs://{gs_path}/ligand_*/abfe.*.INFO* {temp_dir}/',
      shell=True,
      capture_output=True,
      text=True,
  )

  # Create ligand -> log map.
  ligand_to_log = {}
  for log_file_path in log_files:
    # Becuase if it's a restarted job we might have >1 logfiles
    # and we need to identify the latest one.
    ligand = log_file_path.split('/')[-2]
    log_fname = log_file_path.split('/')[-1]
    if ligand not in ligand_to_log.keys():
      ligand_to_log[ligand] = []
    ligand_to_log[ligand].append(log_fname)

  # Resolve which log to use if >1 log files.
  for ligand in ligand_to_log.keys():
    if len(ligand_to_log[ligand]) == 1:
      ligand_to_log[ligand] = ligand_to_log[ligand][0]
    else:
      logs = ligand_to_log[ligand]
      datetimes = [
          int(log.split('.INFO.')[-1].replace('-', '').replace('.', ''))
          for log in logs
      ]
      idx_latest = argmax(datetimes)
      ligand_to_log[ligand] = logs[idx_latest]

  return ligand_to_log, temp_dir


def _job_is_completed(log_file):
  with open(log_file, 'r') as f:
    lines = f.readlines()

  if 'Job completed' in lines[-1]:
    return True
  else:
    return False


def _get_abfe_stage_from_log(log_file):
  with open(log_file, 'r') as f:
    lines = f.read()

  if 'Job completed' in lines:
    return 'completed'

  if 'Analyzing results' in lines:
    return 'analysis'

  if 'Running alchemical transitions' in lines:
    return 'transitions'

  if 'Running production equilibrium simulations' in lines:
    return 'equil sims'

  if 'Running equilibration' in lines:
    return 'equilibration'

  if 'Running energy minimizations' in lines:
    return 'energy minimization'

  if 'Assembling simulation systems' in lines:
    return 'system setup'

  return 'unkwown'


def _get_stage_progress_from_log(log_file):
  with open(log_file, 'r') as f:
    lines = f.read().splitlines()

  num_tpr_run = 0
  num_tpr_total = '?'
  for line in reversed(lines):
    if '-->' in line:
      num_tpr_run += 1
    if 'Running ' in line:
      if '[' in line:
        num_tpr_total = line.split('[')[-1].split(']')[0]
      break

  return f'{num_tpr_run}/{num_tpr_total}'


def _get_running_and_pending_jobs(region='us-central1'):
  process = subprocess.run(
      (
          'gcloud ai custom-jobs list --project="abfe-364520"'
          f' --region={region} --filter="state=JOB_STATE_RUNNING OR'
          ' state=JOB_STATE_PENDING"'
      ),
      shell=True,
      capture_output=True,
      text=True,
  )
  jobs_info = process.stdout.split('---')

  # Add '{protein}/{ligand}' to this list if job running.
  running_jobs = []
  pending_jobs = []

  for job_info in jobs_info:
    lines = job_info.split('\n')
    ligand = None
    protein = None
    state = None
    for line in lines:
      if 'state: ' in line:
        state = line.replace('state: ', '')
      if '--lig_dir' in line:
        ligand = line.split('=')[-1]
      # TODO: find better way to get target name (i.e. output location)
      # This relies on a specific job name structure.
      if 'displayName' in line:
        protein = line.replace('displayName: ', '').split('_')[1]

    job_key = f'{protein}/{ligand}'
    if state == 'JOB_STATE_RUNNING':
      running_jobs.append(job_key)
    elif state == 'JOB_STATE_PENDING':
      pending_jobs.append(job_key)

  return running_jobs, pending_jobs


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  proteins = _parse_protein_str(_PROTEINS.value)
  running_jobs, pending_jobs = _get_running_and_pending_jobs()

  # Table headers.
  s = 90
  print()
  title = '> ' + _OUT_PATH.value + ' <'
  print(f'{title:^{s}}')
  print('=' * s)
  print(
      f'{"Protein":<12}{"Ligand":<30}{"Job State":<15}{"ABFE Stage":<22}{"Progress"}'
  )

  for protein in proteins:
    print('-' * s)
    gs_path_input = os.path.join(_INP_PATH.value, protein)
    gs_path_output = os.path.join(_OUT_PATH.value, protein)

    ligands = _get_ligand_folders(f'gs://{gs_path_input}/ligand_*/')
    ligand_to_log_dict, log_dir = _copy_log_files(gs_path_output)

    for ligand in ligands:
      abfe_stage = ''
      progress = ''
      if ligand in ligand_to_log_dict.keys():
        log_file = os.path.join(log_dir, ligand_to_log_dict[ligand])
        abfe_stage = _get_abfe_stage_from_log(log_file)
        progress = _get_stage_progress_from_log(log_file)
        if f'{protein}/{ligand}' in running_jobs:
          state = f"{_COLORS['white']}RUNNING{_COLORS['endc']}"
        else:
          if _job_is_completed(log_file):
            state = f"{_COLORS['green']}COMPLETED{_COLORS['endc']}"
          else:
            state = f"{_COLORS['red']}CRASHED{_COLORS['endc']}"
      else:
        if f'{protein}/{ligand}' in pending_jobs:
          state = f"{_COLORS['grey']}PENDING{_COLORS['endc']}"
        else:
          state = f"{_COLORS['yellow']}MISSING{_COLORS['endc']}"

      print(f'{protein:<12}{ligand:<30}{state:<24}{abfe_stage:<22}{progress}')

    # rm logs.
    shutil.rmtree(log_dir)

  print('=' * s)
  print()


if __name__ == '__main__':
  app.run(main)
