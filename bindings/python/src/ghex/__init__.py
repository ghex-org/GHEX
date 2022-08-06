# -*- coding: utf-8 -*-
#
# GridTools
#
# Copyright (c) 2014-2021, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
__copyright__ = "Copyright (c) 2014-2021 ETH Zurich"
__license__ = "BSD-3-Clause"

import os
import sys
import warnings

sys.path.append(os.environ.get('GHEX_PY_LIB_PATH', "/home/tille/Development/GHEX/build"))

from ghex.utils.cpp_wrapper_utils import unwrap, CppWrapper
import ghex_py_bindings as _ghex

def _may_use_mpi4py():
    try:
        import mpi4py
        return True
    except:
        return False

def _validate_library_version():
    """check mpi library version string of mpi4py and bindings match"""
    if not _may_use_mpi4py():
        return
    import mpi4py.MPI

    ghex_mpi_lib_ver = _ghex.utils.mpi_library_version()
    mpi4py_lib_ver = mpi4py.MPI.Get_library_version()
    # fix erroneous nullbyte at the end
    if mpi4py_lib_ver[-1] == "\x00":
        mpi4py_lib_ver = mpi4py_lib_ver[:-1]
    if ghex_mpi_lib_ver != mpi4py_lib_ver:
        warnings.warn(f"GHEX and mpi4py were compiled using different mpi versions.\n"
                      f" GHEX:   {ghex_mpi_lib_ver}\n"
                      f" mpi4py: {mpi4py_lib_ver}.")
_validate_library_version()

def make_pattern(context, halo_gen, domain_range):
    # todo: select based on arg types
    return _ghex.make_pattern(unwrap(context), unwrap(halo_gen), [unwrap(d) for d in domain_range])

# note: we don't use the CppWrapper to avoid the runtime overhead
def CommunicationObject(communicator, grid_type: str, domain_id_type: str):
    cls = getattr(_ghex, f"gridtools::ghex::communication_object<{communicator.__cpp_type__}, {grid_type}, {domain_id_type}>")
    return cls(communicator)

#wrap_field = _ghex.wrap_field

#def wrap_field(domain_desc: DomainDescriptor, field: np.ndarray, offsets: Sequence[int, ...], extents: Sequence[int, ...]):
#    FieldDescriptor
#    ghex.wrap_field(domain_desc.__wrapped__,
#                    field,
#                    offsets,
#                    extents)
from .structured.regular import *

from . import tl