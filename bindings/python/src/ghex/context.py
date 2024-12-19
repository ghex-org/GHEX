#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from __future__ import annotations
from typing import TYPE_CHECKING

from ghex import mpi_comm
from ghex.pyghex import context

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

# Compare versions and warn if they differ
import warnings
from mpi4py import __version__ as mpi4py_runtime_version
from ghex import MPI4PY_BUILD_VERSION
if MPI4PY_BUILD_VERSION != mpi4py_runtime_version:
    warnings.warn(
        f"mpi4py version mismatch detected!\n"
        f"Build-time version: {MPI4PY_BUILD_VERSION}\n"
        f"Runtime version: {mpi4py_runtime_version}\n"
        f"This may cause unexpected behavior. Please ensure the versions match.",
        RuntimeWarning,
    )

def make_context(comm: Comm, thread_safe: bool = False) -> context:
    return context(mpi_comm(comm), thread_safe)
