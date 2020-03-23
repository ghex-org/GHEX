ARG BUILD_FROM=ubuntu:19.10
FROM ${BUILD_FROM}
LABEL maintainer="Mauro Bianco <bianco@cscs.ch>"

RUN apt-get update -qq && \
    apt-get install -qq -y \
    build-essential \
    wget \
    git \
    tar \
    software-properties-common && \
    rm -rf /var/lib/apt/lists/*

ARG CMAKE_VERSION=3.14.5
RUN wget -q https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    tar xzf cmake-${CMAKE_VERSION}-Linux-x86_64.tar.gz && \
    cp -r cmake-${CMAKE_VERSION}-Linux-x86_64/bin cmake-${CMAKE_VERSION}-Linux-x86_64/share /usr/local/ && \
    rm -rf cmake-${CMAKE_VERSION}-Linux-x86_64*

ARG BOOST_VERSION=1.67.0
RUN export BOOST_VERSION_UNERLINE=$(echo ${BOOST_VERSION} | sed 's/\./_/g') && \
    wget -q https://dl.bintray.com/boostorg/release/${BOOST_VERSION}/source/boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    tar xzf boost_${BOOST_VERSION_UNERLINE}.tar.gz && \
    cp -r boost_${BOOST_VERSION_UNERLINE}/boost /usr/local/include/ && \
    rm -rf boost_${BOOST_VERSION_UNERLINE}*

RUN  apt-get update && \
     apt-get --assume-yes install openmpi-bin
     
## RUN wget -q http://www.mpich.org/static/downloads/3.1.4/mpich-3.1.4.tar.gz && \
##     tar xf mpich-3.1.4.tar.gz && \
##     cd mpich-3.1.4 && \
##     ./configure --with-device=ch3:sock --disable-fortran --enable-fast=all,O3 --prefix=/usr && \
##     make -j $(nproc) && \
##     make install && \
##     ldconfig && \
##     cd .. && \
##     rm -rf mpich-3.1.4*

ARG GTBENCH_BACKEND=mc
RUN git clone -b release_v1.1 https://github.com/GridTools/gridtools.git && \
    mkdir -p gridtools/build && \
    cd gridtools/build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_TESTING=OFF \
    -DGT_ENABLE_BACKEND_X86=ON \
    -DGT_ENABLE_BACKEND_MC=ON \
    -DMPIEXEC_PREFLAGS=--oversubscribe \
    -DGT_ENABLE_BACKEND_CUDA=OFF \
    -DGT_ENABLE_BACKEND_NAIVE=OFF \
    .. && \
    make -j $(nproc) install && \
    cd ../.. && \
    rm -rf gridtools

COPY . /GHEX
RUN cd /GHEX && \
    mkdir -p build && \
    cd build && \
    cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DGridTools_DIR=/usr/local/lib/cmake \
    -DGHEX_BUILD_TESTS=ON \
    -DMPIEXEC_PREFLAGS="--allow-run-as-root;--oversubscribe" \
    .. && \
    make -j $(nproc) install


RUN cd /GHEX/build && \
    ctest --verbose

CMD ["bash", "-c", "cd /GHEX/build && ctest --verbose"]