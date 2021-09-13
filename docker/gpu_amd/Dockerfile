ARG REPOSITORY=mbianco/ghex-cpu
ARG TAG=gcc-9
FROM ${REPOSITORY}:${TAG}
LABEL maintainer="Mauro Bianco <mbianco@cscs.ch>"

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

# TO DO: xenial (distro code name) still hard-coded
RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | tee /etc/apt/sources.list.d/rocm.list && \
    apt-get update -qq && \
    apt-get install -qq -y rocm-dev && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/opt/rocm/bin:${PATH} CXX=/opt/rocm/llvm/bin/clang++
