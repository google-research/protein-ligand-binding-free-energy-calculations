# Build the image by the following command:
#   docker build -t <image_name>:<tag> --build-arg commit .
FROM nvidia/cuda:11.4.1-devel-ubuntu20.04 AS builder

# Default to merge-release-2023
ARG commit=2e9e6c5abdb395c30a52c699eab3be5c61331bf2

ENV DEBIAN_FRONTEND noninteractive
WORKDIR /root
RUN apt-get update && \
    apt-get install -y --no-install-recommends cmake wget python3 \
    python3-dev python3-pip git

RUN pip3 install --upgrade pip setuptools wheel cmake pybind11

# Checkout the specified Gromacs commit
RUN mkdir -p /root/gromacs
RUN git clone https://gitlab.com/gromacs/gromacs.git
RUN cd gromacs && git checkout ${commit} && \
    mkdir build && cd build && \
    cmake .. -DCMAKE_C_COMPILER=gcc \
             -DCMAKE_CXX_COMPILER=g++ \
             -DGMX_MPI=off \
             -DGMX_GPU=CUDA \
             -DGMX_BUILD_OWN_FFTW=ON \
	     -DGMX_SIMD=AVX2_256 \
	     -DCMAKE_BUILD_TYPE=Release && \
    make -j8 install

FROM nvidia/cuda:11.4.1-runtime-ubuntu20.04
COPY --from=builder /usr/local/gromacs /usr/local/gromacs

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 vim ssh sudo python3 python3-pip curl
RUN apt install python-is-python3

ENV PATH=$PATH:/usr/local/gromacs/bin

RUN useradd -ms /bin/bash gromacs && \
    usermod -aG sudo gromacs
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers

COPY ./entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
