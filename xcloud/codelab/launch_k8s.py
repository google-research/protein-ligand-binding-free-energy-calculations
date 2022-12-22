r"""Launcher for running ABFE codelab with a --xm_gcs_path arg.

This script allows external user to launch a XManager job on Google Cloud project to run
GROMACS molecular dynamics simulation. The benchmark simulation files and Docker image
are prepared in Google Cloud bucket as an illustrative example.

Pass the Google Cloud bucket path with the flag --xm_gcs_path.

Usage:

Prerequisites:
1) Install XManager with `pip install git+https://github.com/deepmind/xmanager.git`
2) Create a GCP project and install `gcloud` (https://cloud.google.com/sdk/docs/install)
3) Associate your Google Account (Gmail account) with your GCP project by running:

```
export GCP_PROJECT=<GCP PROJECT ID>
gcloud auth login
gcloud auth application-default login
gcloud config set project $GCP_PROJECT
```

4) Set up gcloud to work with Docker by running `gcloud auth configure-docker`
5) Enable Google Cloud Platform APIs.
* Enable IAM.
* Enable the 'Cloud AI Platfrom'.
* Enable the 'Container Registry'.
* Enable the 'Vertex AI'.

6) Create a staging bucket if you do not already have one. Run the following command

`export GOOGLE_CLOUD_BUCKET_NAME=<GOOGLE_CLOUD_BUCKET_NAME>`

7) Upload a TPR file to the staging bucket, for example to the path 
${GOOGLE_CLOUD_BUCKET_NAME}/${USER}/benchmark.tpr, run the following command.

xmanager launch \
  xcloud/codelab/launch_external.py -- \
    --xm_gcs_path="/gcs/${GOOGLE_CLOUD_BUCKET_NAME}/${USER}" \
    --tpr_file=benchmark.tpr
"""

from absl import app
from absl import flags
from absl import logging

from kubernetes import client as k8s_client
from kubernetes import config as k8s_config
from xmanager import xm
from xmanager.xm import utils
from xmanager import xm_local
from xmanager.xm_local import executables as local_executables
from xmanager.cloud import kubernetes
from xmanager.cloud.kubernetes import convert_to_valid_label
from xmanager.cloud.kubernetes import requirements_from_executor
from xmanager.cloud.kubernetes import node_selector_from_executor
from xmanager.cloud.kubernetes import annotations_from_executor
from xmanager.contrib import gcs


_GPU_TYPE = flags.DEFINE_string('gpu_type', 't4',
                                'Which GPU type to use, e.g., t4, v100, a100.')
_REPLICAS = flags.DEFINE_integer('replicas', 1, 'Task replicas per job.')
_EXP_NAME = flags.DEFINE_string(
    'exp_name', 'abfe', 'Name of the experiment.', short_name='n')
_TPR_FILE = flags.DEFINE_string('tpr_file', None, 'tpr file')
_NUM_CPU = flags.DEFINE_integer('num_cpu', 4, 'The number of CPUs.')
_NUM_MPI = flags.DEFINE_integer('num_mpi', 1,
                                'The number of thread-MPI processes.')
_NUM_THREADS = flags.DEFINE_integer('num_threads', 4,
                                    'The number of OpenMP threads.')
_N_STEPS = flags.DEFINE_integer('n_steps', -2,
                                'The number of simulations steps.')
# This flag is useful for benchmarking.
_NUM_REPLICAS = flags.DEFINE_integer('n_replicas', 1,
                                     'The number of replica runs.')

flags.adopt_module_key_flags(gcs)  # Registers flag --xm_gcs_path.


# Monkey patch the cloud kubernetes launch functions
def _requirements_from_executor(executor):
  requirement = requirements_from_executor(executor)
  requirement.requests = requirement.limits.copy()
  requirement.requests['cpu'] = str(float(requirement.requests['cpu']) - 1.0)
  return requirement


def _launch(
    self,
    get_full_job_name,
    jobs,
):
  """Launches jobs on Kubernetes."""
  batch_jobs = []
  service = 'experiments'
  for job in jobs:
    executable = job.executable
    executor = job.executor
    if not isinstance(executable,
                      local_executables.GoogleContainerRegistryImage):
      raise ValueError('Executable {} has type {}. Executable must be of '
                       'type GoogleContainerRegistryImage.'.format(
                           executable, type(executable)))
    all_env_vars = {**executable.env_vars, **job.env_vars}
    env = [k8s_client.V1EnvVar(k, v) for k, v in all_env_vars.items()]
    job_name = kubernetes.convert_to_valid_label(get_full_job_name(job.name))
    container = k8s_client.V1Container(
        name=job_name,
        image=executable.image_path,
        resources=_requirements_from_executor(executor),
        args=xm.merge_args(executable.args,
                           job.args).to_list(utils.ARG_ESCAPER),
        env=env,
        volume_mounts=[
            k8s_client.V1VolumeMount(name="gcsfuse", mount_path='/gcs')
        ])
    k8s_job = k8s_client.V1Job()
    k8s_job.metadata = k8s_client.V1ObjectMeta(name=job_name)
    k8s_job.spec = k8s_client.V1JobSpec(
        template=k8s_client.V1PodTemplateSpec(
            metadata=k8s_client.V1ObjectMeta(
                labels={'service': service},
                annotations=annotations_from_executor(executor),
            ),
            spec=k8s_client.V1PodSpec(
                service_account=kubernetes._K8S_SERVICE_ACCOUNT_NAME.value,
                hostname=job_name,
                subdomain=service,
                restart_policy='Never',
                containers=[container],
                node_selector=node_selector_from_executor(executor),
                volumes=[
                    k8s_client.V1Volume(
                        name='gcsfuse',
                        persistent_volume_claim=k8s_client.V1PersistentVolumeClaimVolumeSource(
                            claim_name='gcsfuse-csi-pvc'
                        ))
                ]
            ),
        ),
        backoff_limit=0,
    )
    batch_jobs.append(k8s_job)

  self._create_service(service)
  batch_api = k8s_client.BatchV1Api(self.api_client)
  for k8s_job in batch_jobs:
    batch_api.create_namespaced_job(namespace='default', body=k8s_job)
  return batch_jobs


kubernetes.Client.launch = _launch


def _gromacs_pkgs() -> xm.Packageable:
  """Returns a contructed python container for running Gromacs simulations.
  """
  workdir = gcs.get_gcs_path_or_fail(_EXP_NAME.value)
  # Experiment expects output dir in '/gcs/...' format.
  # Might not needed: workdir = gcs.get_gcs_fuse_path(workdir)
  executable_args = {
      'data_dir': workdir,
      'tpr_file': _TPR_FILE.value,
      'num_mpi': _NUM_MPI.value,
      'num_threads': _NUM_THREADS.value,
      'n_steps': _N_STEPS.value,
  }
  return [
      xm.python_container(
          executor_spec=xm_local.Kubernetes.Spec(),
          args=executable_args,
          base_image='gcr.io/abfe-364520/gromacs/gromacs-base:cuda-11.4-avx2_256',
          entrypoint=xm.ModuleName('codelab.main'),
          use_deep_module=True,
      )
  ]


def main(argv) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  if _NUM_CPU.value != _NUM_THREADS.value * _NUM_MPI.value:
    logging.error(
        'Number of total threads does NOT equal to the number of CPUs. '
        'Performance might suffer.'
    )

  with xm_local.create_experiment(experiment_title=_EXP_NAME.value) as experiment:
    [executable] = experiment.package(_gromacs_pkgs())

    job_requirements = xm.JobRequirements({xm.GpuType[_GPU_TYPE.value]: 1},
                                          cpu=_NUM_CPU.value,
                                          ram=1 * xm.GiB)
    executor = xm_local.Kubernetes(requirements=job_requirements)

    for i in range(_NUM_REPLICAS.value):
      experiment.add(xm.Job(executable, executor, args={'replica_id': str(i)}))


if __name__ == '__main__':
  app.run(main)
