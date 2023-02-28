# Binding free energy calculations pipeline on GCP with Gromacs

This repository will contain a pipeline developed by Google Research, the Max Planck Institute for Multidisciplinary Sciences, and Janssen, to carry out protein-ligand binidng free energy calculations using [Gromacs](https://www.gromacs.org) and [Google Cloud Platform](https://cloud.google.com) (GCP).

## Disclaimer

This is not an officially supported Google product. The repo is also not under active development.

## Content

- `colab_tutorial` contains Colab notebooks with simple examples on how to run Gromacs and binding free energy calculations in Colab, as well as how to use [xmanager](https://github.com/deepmind/xmanager) to submit jobs in GCP.
- `docker` contains files that can be used to build Docker images with Gromacs and [pmx](https://github.com/deGrootLab/pmx/tree/abfe_dev).
- `xcloud` contains example of how to run simulations, as well as more complex absolute binding free energy (ABFE)calculation protocols, on Google Cloud using [Vertex AI](https://cloud.google.com/vertex-ai).

## Usage
The main script defining the ABFE protocol is `xcloud/abfe_base_protocol/main.py`. This is based on a non-equilibriunm protocol, similar to that proposed by [Khalak et al.](https://pubs.rsc.org/en/content/articlelanding/2021/sc/d1sc03472c). 

A job can be submitted to GCP with the script `xcloud/abfe_base_protocol/launch_vertex.py` and using [xmanager](https://github.com/deepmind/xmanager). To be able to submit the calculations on GCP, you need to have a [GCP project](https://cloud.google.com/resource-manager/docs/creating-managing-projects) and billing account set up. In addition, a docker image with both Gromacs and the `pmx` Python library should be available in the [Artifact Registry](https://cloud.google.com/artifact-registry).

Once all the above is ready, a calculation (or a batch of calculations) can be submitted as follows:

```
xmanager launch launch_vertex.py -- \
  --base_image="gcr.io/$MY_PROJECT/$PATH_TO_DOCKER_IMAGE" \
  --top_path="/$GCS_PATH_TO_TOP_FILES" \
  --mdp_path="/$GCS_PATH_TO_MDP_FILES" \
  --out_path="/$GCS_OUTPUT_PATH" \
  --lig_ids="1,10"
```

Where `base_image` refers to the Docker image, and `top_path`, `mdp_path`, and `out_path` refer to paths in [Cloud buckets](https://cloud.google.com/storage/docs/creating-buckets). For additional information, see the description of the script's arguments.