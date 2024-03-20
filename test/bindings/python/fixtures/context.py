#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
import mpi4py
from mpi4py import MPI
import pytest

from ghex.context import make_context


mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"


@pytest.fixture
def context():
    return make_context(MPI.COMM_WORLD, True)


@pytest.fixture
def mpi_cart_comm():
    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), [0, 0, 0])
    mpi_cart_comm = mpi_comm.Create_cart(dims=dims, periods=[False, False, False])
    return mpi_cart_comm


@pytest.fixture
def cart_context(mpi_cart_comm):
    return make_context(mpi_cart_comm, True)
