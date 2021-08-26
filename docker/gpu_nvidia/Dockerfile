ARG REPOSITORY=mbianco/ghex-cpu
ARG TAG=gcc-9
FROM ${REPOSITORY}:${TAG}
LABEL maintainer="Mauro Bianco <mbianco@cscs.ch>"

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

# TO DO: ubuntu release (20.04) and cuda version (11.3) are still hard-coded
RUN apt-get update -qq && \
    wget -q -O - https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    apt-get install -qq -y cuda-minimal-build-11-3 && \
    rm -rf /var/lib/apt/lists/*

ENV CUDACXX=/usr/local/cuda-11.3/bin/nvcc CUDAHOSTCXX=${CXX}
