ARG DEPS_IMAGE
FROM $DEPS_IMAGE

COPY . /ghex
WORKDIR /ghex

ARG BACKEND
ARG NUM_PROCS
RUN spack -e ci build-env ghex -- \
        cmake -G Ninja -B build \
            -DCMAKE_BUILD_TYPE=Debug \
            -DGHEX_WITH_TESTING=ON \
            -DGHEX_TRANSPORT_BACKEND=$(echo $BACKEND | tr '[:lower:]' '[:upper:]') \
            -DGHEX_USE_BUNDLED_LIBS=ON \
            -DGHEX_USE_BUNDLED_OOMPH=OFF \
            -DGHEX_USE_BUNDLED_GRIDTOOLS=OFF \
            -DGHEX_USE_GPU=ON \
            -DGHEX_GPU_TYPE=NVIDIA \
            -DMPIEXEC_EXECUTABLE="" \
            -DMPIEXEC_NUMPROC_FLAG="" \
            -DMPIEXEC_PREFLAGS="" \
            -DMPIEXEC_POSTFLAGS="" && \
    spack -e ci build-env ghex -- cmake --build build -j$NUM_PROCS
