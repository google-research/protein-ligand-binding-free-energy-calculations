# Build the image by the following command:
#   docker build -t <image_name>:<tag> .
FROM gcr.io/abfe-364520/gromacs/gromacs-base:cuda-11.4-avx2_256

RUN apt-get update && \
    apt-get install -y python3-dev build-essential git
RUN pip3 install git+https://github.com/deGrootLab/pmx.git@abfe_dev 
