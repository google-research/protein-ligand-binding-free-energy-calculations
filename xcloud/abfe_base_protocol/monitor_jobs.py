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

r"""Script to summarize the status of ABFE calculations running on GCP.
"""


from typing import List
import subprocess
import os
import shutil
import tempfile

from absl import app
from absl import flags


# Flags about paths.
_INP_PATH = flags.DEFINE_string('input_path', None, 'GCS path to input files for calculations.')
_OUT_PATH = flags.DEFINE_string('output_path', None, 'GCS path to output files of calculations.')
_PROTEINS = flags.DEFINE_string('proteins', None, 'Folder names of target proteins (comma-separated list, e.g. "cmet,jnk1").')

# Required flag.
flags.mark_flag_as_required("input_path")
flags.mark_flag_as_required("output_path")
flags.mark_flag_as_required("proteins")

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
    'endc': '\033[0m'
}


def _parse_protein_str(protein_str: str) -> List:
  return [i for i in protein_str.split(',')]


def _get_ligand_folders(gs_path):
  process = subprocess.run(f'gsutil ls -d {gs_path}',
                             shell=True, capture_output=True, text=True)
  ligands = [d.split('/')[-2] for d in process.stdout.split('\n') if d]
  return ligands


def _copy_log_files(gs_path):
  # List all logs.
  process = subprocess.run(f'gsutil ls gs://{gs_path}/ligand_*/abfe.*.INFO*',
                             shell=True, capture_output=True, text=True)
  log_files = [f for f in process.stdout.split('\n') if f]
  
  # If empty just return empty dict.
  if len(log_files) < 1:
    return {}

  # Copy log files.
  temp_dir = tempfile.mkdtemp()
  process = subprocess.run(f'gsutil -m cp gs://{gs_path}/ligand_*/abfe.*.INFO* {temp_dir}/',
                             shell=True, capture_output=True, text=True)

  # Create ligand -> log map.
  ligand_to_log = {}
  for log_file_path in log_files:
    ligand = log_file_path.split('/')[-2]
    log_fname = log_file_path.split('/')[-1]
    ligand_to_log[ligand] = log_fname

  return ligand_to_log, temp_dir


def _get_status_from_log(log_file):
  with open(log_file, 'r') as f:
    lines = f.readlines()
  
  if 'completed' in lines[-1]:
    return f"{_COLORS['green']}COMPLETED{_COLORS['endc']}"
  else:
    return f"{_COLORS['blue']}RUNNING (OR CRASHED){_COLORS['endc']}"


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  proteins = _parse_protein_str(_PROTEINS.value)
  
  for protein in proteins:
    gs_path_input = os.path.join(_INP_PATH.value, protein)
    gs_path_output = os.path.join(_OUT_PATH.value, protein)

    ligands = _get_ligand_folders(f'gs://{gs_path_input}/ligand_*/')
    ligand_to_log_dict, log_dir = _copy_log_files(gs_path_output)

    s = 60
    print('=' * s)
    name = gs_path_input.split('/')[-1]
    print(name.center(s))
    print('=' * s)
    for ligand in ligands:
      if ligand in ligand_to_log_dict.keys():
        log_file = os.path.join(log_dir, ligand_to_log_dict[ligand])
        status = _get_status_from_log(log_file)
      else:
        status = f"{_COLORS['grey']}MISSING{_COLORS['endc']}"

      print(f"{ligand:<40}{status}")
    print('=' * s)
    print('')

    # rm logs.
    shutil.rmtree(log_dir)

if __name__ == '__main__':
  app.run(main)