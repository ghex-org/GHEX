[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![CI](https://github.com/ghex-org/GHEX/actions/workflows/CI.yml/badge.svg)](https://github.com/ghex-org/GHEX/actions/workflows/CI.yml)
[![Pip](https://github.com/ghex-org/GHEX/actions/workflows/test_pip.yml/badge.svg)](https://github.com/ghex-org/GHEX/actions/workflows/test_pip.yml)
# GHEX
Generic exascale-ready library for halo-exchange operations on variety of grids/meshes.

Documentation and instructions at [GHEX Documentation](https://ghex-org.github.io/GHEX/).

### Installation instructions


#### Pip Install

```
python -m venv ghex_venv
. ghex_venv/bin/activate
python -m pip install ghex
```

##### Pertinent environment variables

| Variable                  | Allowed Values          | Default                            | Description              |
| ------------------------- | ----------------------- | ---------------------------------- | ------------------------ |
| `GHEX_USE_GPU=`           | `{ON, OFF}`             | `OFF`                              | Enable GPU               |
| `GHEX_GPU_TYPE=`          | `{AUTO, NVIDIA, AMD}`   | `AUTO`                             | Choose GPU type          |
| `GHEX_GPU_ARCH=`          | list of archs           | `"60;70;75;80"`/ `"gfx900;gfx906"` | GPU architecture         |
| `GHEX_TRANSPORT_BACKEND=` | `{MPI, UCX, LIBFABRIC}` | `MPI`                              | Choose transport backend |

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
