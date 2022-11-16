r"""The main program for running gromacs simulation with a TPR input.

This module is simply for testing running gromacs with XCloud and it is subject
to modification.
"""

import os
import time

from absl import app
from absl import flags
from absl import logging

import gmxapi as gmx

_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data directory')
_TPR_FILE = flags.DEFINE_string('tpr_file', None, 'tpr file')
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


gmx.operation.ResourceManager.operation_id = property(operation_id)


def main(_):
  logging.info('Job started.')

  if not _DATA_DIR.value:
    logging.fatal('Data directory not specified.')
  if not _TPR_FILE.value:
    logging.fatal('TPR file not specified.')
  if not _TPR_FILE.value.endswith('.tpr'):
    logging.fatal('TPR file must end with .tpr extension.')

  logging.info('Change workdir to %s', _DATA_DIR.value)
  os.chdir(_DATA_DIR.value)
  input_tpr = gmx.read_tpr(_TPR_FILE.value)
  md = gmx.mdrun(
      input_tpr,
      runtime_args={
          '-nb': 'gpu',
          '-pme': 'gpu',
          '-pmefft': 'gpu',
          '-bonded': 'gpu',
          '-deffnm': os.path.splitext(_TPR_FILE.value)[0],
          '-ntmpi': str(_NUM_MPI.value),
          '-ntomp': str(_NUM_THREADS.value),
          '-pin': 'on'
      })

  md.run()
  logging.info('Job finished.')


if __name__ == '__main__':
  app.run(main)
  