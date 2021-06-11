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

ARG CMAKE_VERSION=3.14.5
RUN cd /tmp && \
    wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/bin cmake-${CMAKE_VERSION}-Linux-x86_64/share /usr/local/ && \
    rm -rf *

# Boost temporarily installed from ubuntu repositories
# due to bintray repository unavailable (403 - forbidden)
#ARG BOOST_VERSION=1.70.0
#RUN cd /tmp && \
#    export BOOST_VERSION_UNDERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
#    wget -q https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNDERLINE}.tar.gz && \
#    tar xzf boost_${BOOST_VERSION_UNDERLINE}.tar.gz && \
#    cp -r boost_${BOOST_VERSION_UNDERLINE}/boost /usr/local/include/ && \
#    rm -rf *
RUN apt-get update -qq && \
    apt-get install -qq -y libboost-all-dev && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    apt-get install -qq -y libtool automake autoconf && \
    rm -rf /var/lib/apt/lists/*

RUN apt-get update -qq && \
    apt-get install -qq -y libnuma-dev && \
    rm -rf /var/lib/apt/lists/*

RUN cd /tmp && \
    git clone https://github.com/openucx/ucx.git && \
    cd ucx && \
    git checkout v1.10.1 && \
    ./autogen.sh && \
    ./configure --disable-logging --disable-debug --disable-assertions --disable-params-check --enable-optimizations --enable-compiler-opt=3 --enable-mt --enable-devel-headers --without-java --prefix=/opt/ucx && \
    make -j$(nproc) && make install && \
    cd /tmp && \
    rm -rf ucx

ENV UCX_TLS=self,tcp

RUN cd /tmp && \
    wget -q https://download.open-mpi.org/release/open-mpi/v4.1/openmpi-4.1.1.tar.gz && \
    tar xzf openmpi-4.1.1.tar.gz && \
    cd openmpi-4.1.1 && \
    ./configure --with-hwloc=internal --with-ucx=/opt/ucx --prefix=/opt/openmpi_ucx && \
    make all install && \
    cd /tmp && \
    rm -rf openmpi-4.1.1.tar.gz openmpi-4.1.1

ENV MPIROOT=/opt/openmpi_ucx
ENV PATH=${MPIROOT}/bin:${PATH} LD_LIBRARY_PATH=${MPIROOT}/lib:${LD_LIBRARY_PATH} PKG_CONFIG_PATH=${MPIROOT}/lib/pkgconfig:${PKG_CONFIG_PATH}

RUN cd /opt && \
    git clone https://github.com/ecmwf/ecbuild.git

RUN cd /tmp && \
    git clone https://github.com/ecmwf/eckit.git && \
    cd eckit && \
    git checkout master && \
    mkdir -p build && cd build && \
    /opt/ecbuild/bin/ecbuild --build=RELEASE --prefix=/opt/eckit -- ../ && \
    make -j$(nproc) install && \
    cd /tmp && \
    rm -rf eckit

RUN cd /tmp && \
    git clone https://github.com/ecmwf/atlas.git && \
    cd atlas && \
    git checkout master && \
    mkdir -p build && cd build && \
    export ECKIT_PATH=/opt/eckit && \
    /opt/ecbuild/bin/ecbuild --build=RELEASE --prefix=/opt/atlas -- ../ && \
    make -j$(nproc) install && \
    cd /tmp && \
    rm -rf atlas

RUN cd /tmp && \
    wget -q http://glaros.dtc.umn.edu/gkhome/fetch/sw/parmetis/parmetis-4.0.3.tar.gz && \
    tar xzf parmetis-4.0.3.tar.gz && \
    cd parmetis-4.0.3 && \
    sed -i 's/#define IDXTYPEWIDTH 32/#define IDXTYPEWIDTH 64/g' metis/include/metis.h && \
    cd metis && \
    make config shared=1 prefix=/opt/metis/ && \
    make install && \
    cd .. && \
    make config shared=1 prefix=/opt/parmetis/ && \
    make install && \
    cd /tmp && \
    rm -rf parmetis-4.0.3.tar.gz parmetis-4.0.3

#RUN useradd -ms /bin/bash github
#
#USER github
#WORKDIR /home/github
