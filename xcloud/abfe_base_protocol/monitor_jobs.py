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


def _parse_protein_str(protein_str: str) -> List:
  return [i for i in protein_str.split(',')]


def _get_ligand_folders(gs_path):
  process = subprocess.run(f'gsutil ls -d {gs_path}',
                             shell=True, capture_output=True, text=True)
  ligands = [d.split('/')[-2] for d in process.stdout.split('\n') if d]
  return ligands


def _get_log_file(gs_path, ligand):
  process = subprocess.run(f'gsutil ls -d gs://{gs_path}/{ligand}/abfe.*.INFO*',
                             shell=True, capture_output=True, text=True)
  log_file = [f for f in process.stdout.split('\n') if f]
  if len(log_file) < 1:
    return ''
  elif len(log_file) == 1:
    return log_file[0]
  else:
    raise ValueError()


def _get_status_from_log(log_file):
  if not log_file:
    return '...'

  process = subprocess.run(f'gsutil cat {log_file} | tail -n 1',
                           shell=True, capture_output=True, text=True)
  if 'completed' in process.stdout:
    return 'COMPLETED'
  else:
    return 'RUNNING'


def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  proteins = _parse_protein_str(_PROTEINS.value)
  
  for protein in proteins:
    gs_path_input = os.path.join(_INP_PATH.value, protein)
    gs_path_output = os.path.join(_OUT_PATH.value, protein)

    ligands = _get_ligand_folders(f'gs://{gs_path_input}/ligand_*/')

    print('=' * 50)
    name = gs_path_input.split('/')[-1]
    print(name.center(50))
    print('=' * 50)
    for ligand in ligands:
      log_file = _get_log_file(gs_path_output, ligand)
      status = _get_status_from_log(log_file)
      print(f"{ligand:<40}{status}")
    print('=' * 50)
    print('')


if __name__ == '__main__':
  app.run(main)