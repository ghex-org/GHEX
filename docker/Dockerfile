FROM ubuntu:latest
LABEL maintainer="Mauro Bianco <mbianco@cscs.ch>"

SHELL ["/bin/bash", "-lc"]

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update -qq && \
    apt-get install -qq -y \
        build-essential \
        ccache \
        wget \
        git \
        tar \
        software-properties-common \
        lsb-release \
        cowsay && \
    ln -s /usr/games/cowsay /usr/bin/cowsay && \
    rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.14.5
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/bin cmake-${CMAKE_VERSION}-Linux-x86_64/share /usr/local/ && \
    rm -rf *

ARG BOOST_VERSION=1.70.0
RUN cd /tmp && \
    export BOOST_VERSION_UNDERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
    wget -q https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERLINE}.tar.gz && \
    tar xzf boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    cp -r boost_${BOOST_VERSION_UNERLINE}/boost /usr/local/include/ && \
    rm -rf *

RUN apt-get update -qq && \
    apt-get install -qq -y openmpi-bin libopenmpi-dev && \
    rm -rf /var/lib/apt/lists/*

ARG GCC_VERSION=9
ENV GCC_VERSION ${GCC_VERSION}

RUN apt-get update -qq && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get install -qq -y \
        g++-${GCC_VERSION} \
        gcc-${GCC_VERSION} \
        gfortran-${GCC_VERSION} && \
    rm -rf /var/lib/apt/lists/*

ENV CXX=g++-${GCC_VERSION} CC=gcc-${GCC_VERSION} FC=gfortran-${GCC_VERSION}

RUN git clone https://github.com/ecmwf/ecbuild.git

RUN git clone https://github.com/ecmwf/eckit.git && \
    cd eckit && \
    git checkout master && \
    mkdir -p build && cd build && \
    /ecbuild/bin/ecbuild --build=RELEASE --prefix=/opt/eckit -- ../ && \
    make install

RUN git clone https://github.com/ecmwf/atlas.git && \
    cd atlas && \
    git checkout master && \
    mkdir -p build && cd build && \
    export ECKIT_PATH=/opt/eckit && \
    /ecbuild/bin/ecbuild --build=RELEASE --prefix=/opt/atlas -- ../ && \
    make install

RUN apt-get update -qq && \
    apt-get install -qq -y libtool automake autoconf && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    apt-get install -qq -y libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/openucx/ucx.git && \
    cd ucx && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local --enable-mt --disable-numa && \
    make && make install

ENV LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
ENV UCX_TLS=self,tcp

# HERE

#RUN apt-get update -qq && \
#    wget -q -O- https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub | apt-key add - && \
#    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/cuda.list && \
#    echo "deb https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64 /" > /etc/apt/sources.list.d/nvidia-ml.list && \
#    rm -rf /var/lib/apt/lists/*

#RUN apt-get update -qq && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y cuda-minimal-build-11-1 && \
#    rm -rf /var/lib/apt/lists/*

#ENV CUDACXX=/usr/local/cuda-11.1/bin/nvcc CUDAHOSTCXX=${CXX}

#RUN wget -q -O - https://repo.radeon.com/rocm/rocm.gpg.key | apt-key add - && \
#    echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian/ xenial main' | tee /etc/apt/sources.list.d/rocm.list && \
#    apt-get update -qq && \
#    DEBIAN_FRONTEND=noninteractive apt-get install -qq -y rocm-dev && \
#    rm -rf /var/lib/apt/lists/*

#ENV ROCM_PATH=/opt/rocm
#ENV PATH=${ROCM_PATH}/bin:${PATH} CXX=${ROCM_PATH}/bin/hipcc
