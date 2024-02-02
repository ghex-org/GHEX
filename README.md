[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/ghex-org/GHEX/actions/workflows/CI.yml/badge.svg)](https://github.com/ghex-org/GHEX/actions/workflows/CI.yml)
[![Pip](https://github.com/ghex-org/GHEX/actions/workflows/test_pip.yml/badge.svg)](https://github.com/ghex-org/GHEX/actions/workflows/test_pip.yml)
# GHEX
Generic exascale-ready library for halo-exchange operations on variety of grids/meshes.

Documentation and instructions at [GHEX Documentation](https://ghex-org.github.io/GHEX/).

### Installation instructions

#### Requirements
- C++17 compiler (gcc or clang)
- CMake (3.21 or later)
- GridTools (available as submodule with `GHEX_USE_BUNDLED_LIBS=ON`)
- Boost headers (1.80 or later)
- MPI
- UCX (optional)
- Libfabric (optional)
- Xpmem (optional)
- oomph (0.3.0 or later, available as submodule  with `GHEX_USE_BUNDLED_LIBS=ON`)
- hwmalloc (0.3.0 or later, available as submodule  with `GHEX_USE_BUNDLED_LIBS=ON`)
- CUDA Toolkit (optional, 11.0 or later)
- Rocm/Hip Toolkit (optional, 4.5.1 or later)
- Google Test (only when `GHEX_WTIH_TESTING=ON`, available as submodule with `GHEX_USE_BUNDLED_LIBS=ON`)
- parmetis, metis (optional)
- atlas (optional)
- gfortran compiler (optional, for Fortran bindings)
- Python3 (optional, for Python bindings )
- MPI4PY (optional, for Python bindings )
- Pybind11 (optional, for Python bindings)

#### From Source

```
git clone https://github.com/ghex-org/GHEX.git
cd GHEX
git submodule update --init --recursive
mkdir -p build && cd build
cmake -DGHEX_WITH_TESTING=ON -DGHEX_USE_BUNDLED_LIBS=ON ..
make -j8
make test
```

##### CMake options

| Option | Allowed Values | Default | Description |
| --- | --- | --- | --- |
| `CMAKE_INSTALL_PREFIX=`           | `<path>`                | `/usr/local`                                          | Choose install path prefix
| `CMAKE_BUILD_TYPE=`               | `{Release, Debug}`      | `Release`                                             | Choose type of build
| `GHEX_USE_BUNDLED_LIBS=`          | `{ON, OFF}`             | `OFF`                                                 | Use available git submodules
| `GHEX_USE_GPU=`                   | `{ON, OFF}`             | `OFF`                                                 | Enable GPU
| `GHEX_GPU_TYPE=`                  | `{AUTO, NVIDIA, AMD}`   | `AUTO`                                                | Choose GPU type
| `CMAKE_CMAKE_CUDA_ARCHITECTURES=` | list of architectures   | `"60;70;75"`                                          | Only relevant for GHEX_GPU_TYPE=NVIDIA
| `CMAKE_CMAKE_HIP_ARCHITECTURES=`  | list of architectures   | `"gfx900;gfx906"`                                     | Only relevant for GHEX_GPU_TYPE=AMD
| `GHEX_BUILD_FORTRAN=`             | `{ON, OFF}`             | `OFF`                                                 | Build with Fortran bindings
| `GHEX_BUILD_PYTHON_BINDINGS=`     | `{ON, OFF}`             | `OFF`                                                 | Build with Python bindings
| `GHEX_PYTHON_LIB_PATH=`           | `<path>`                | `${CMAKE_INSTALL_PREFIX}/<python-site-packages-path>` | Installation directory for GHEX's Python package
| `GHEX_WITH_TESTING=`              | `{ON, OFF}`             | `OFF`                                                 | Build unit tests
| `GHEX_USE_XPMEM=`                 | `{ON, OFF}`             | `OFF`                                                 | Use Xpmem
| `GHEX_TRANSPORT_BACKEND=`         | `{MPI, UCX, LIBFABRIC}` | `MPI`                                                 | Choose transport backend

#### Pip Install

```
python -m venv ghex_venv
. ghex_venv/bin/activate
python -m pip install 'git+https://github.com/ghex-org/GHEX.git#subdirectory=bindings/python'
```

##### Pertinent environment variables

| Variable | Allowed Values | Default | Description |
| --- | --- | --- | --- |
| `GHEX_USE_GPU=`           | `{ON, OFF}`             | `OFF`                              | Enable GPU
| `GHEX_GPU_TYPE=`          | `{AUTO, NVIDIA, AMD}`   | `AUTO`                             | Choose GPU type
| `GHEX_GPU_ARCH=`          | list of archs           | `"60;70;75;80"`/ `"gfx900;gfx906"` | GPU architecture
| `GHEX_TRANSPORT_BACKEND=` | `{MPI, UCX, LIBFABRIC}` | `MPI`                              | Choose transport backend

### Acknowledgements

The development of GHEX was supported partly by The Partnership for Advanced
Computing in Europe (PRACE). PRACE is an international non-profit association
with its seat in Brussels. The PRACE Research Infrastructure provides a
persistent world-class high performance computing service for scientists and
researchers from academia and industry in Europe. The computer systems and
their operations accessible through PRACE are provided by 5 PRACE members (BSC
representing Spain, CINECA representing Italy, ETH Zurich/CSCS representing
Switzerland, GCS representing Germany and GENCI representing France). The
Implementation Phase of PRACE receives funding from the EU’s Horizon 2020
Research and Innovation Programme (2014-2020) under grant agreement 823767. For
more information, see www.prace-ri.eu.

