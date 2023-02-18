r"""The main program for running gromacs simulation with a TPR input.

This module is simply for testing running gromacs with XCloud and it is subject
to modification.
"""

import os
import time

from absl import app
from absl import flags
from absl import logging

from logging import DEBUG

import gmxapi as gmx
from gmxapi import logger


mdrun_logger = logger.getChild('mdrun')
mdrun_logger.setLevel(DEBUG)


_DATA_DIR = flags.DEFINE_string('data_dir', None, 'data directory')
_TPR_FILE = flags.DEFINE_string('tpr_file', None, 'tpr file')
_NUM_MPI = flags.DEFINE_integer(
    'num_mpi', 1, 'The number of thread-MPI processes.'
)
_NUM_THREADS = flags.DEFINE_integer(
    'num_threads', 4, 'The number of OpenMP threads.'
)
_N_STEPS = flags.DEFINE_integer(
    'n_steps', -2, 'The number of simulations steps.'
)
_UPDATE = flags.DEFINE_string(
    'update',
    'auto',
    'argument for mdrun -update flag; choices are "auto", "cpu", "gpu".',
)
_PIN = flags.DEFINE_string(
    'pin',
    'auto',
    'argument for mdrun -pin flag; choices are "auto", "on", "off".',
)
_REP_ID = flags.DEFINE_integer(
    'replica_id', 0, 'Hyperparam that identifies the replica of this run.'
)


def main(_):
  # Sets a session ID for this session.
  _SESSION_ID = str(int(time.time())) + '_' + str(_REP_ID.value)

  # Monkey patch this session ID into the gmxapi.operation.ResourceManager.
  # Hack. Do not use in the user code.
  # This is temporarily added to allow users to start from a new set of working
  # directories for all the mdrun launches. GMX API doesn't expose the
  # underlying resource manager or context yet.
  def operation_id(self):
    # pylint: disable=protected-access
    return f'{self._base_operation_id}_{_SESSION_ID}'

  gmx.operation.ResourceManager.operation_id = property(operation_id)

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
          '-update': _UPDATE.value,
          '-deffnm': os.path.splitext(_TPR_FILE.value)[0],
          '-ntmpi': str(_NUM_MPI.value),
          '-ntomp': str(_NUM_THREADS.value),
          '-pin': _PIN.value,
          '-nsteps': str(_N_STEPS.value),
      },
  )

  md.run()
  logging.info('Job finished.')


if __name__ == '__main__':
  app.run(main)
