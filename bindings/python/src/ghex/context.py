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


def make_context(comm: Comm, thread_safe: bool = False) -> context:
    return context(mpi_comm(comm), thread_safe)
