# -*- coding: utf-8 -*-
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

from ghex import context as _context
from ghex import mpi_comm as _mpi_comm

if TYPE_CHECKING:
    from mpi4py.MPI import Comm

def make_context(mpi_comm: Comm, thread_safe: bool = False) -> _context:
    return _context(_mpi_comm(mpi_comm), thread_safe)
