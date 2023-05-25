#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
#import pytest

# import the (autouse) fixture which adds the pyghex module to sys.path
from fixtures.pyghex_path import *

# import mpi4py
import mpi4py
mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
mpi4py.rc.threads=True
mpi4py.rc.thread_level="multiple"
from mpi4py import MPI

import ghex

@pytest.fixture
def context():
    comm = ghex.mpi_comm(MPI.COMM_WORLD)
    ctx = ghex.context(comm, True)
    return ctx

@pytest.fixture
def mpi_cart_comm():
    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), [0, 0, 0])
    mpi_cart_comm = mpi_comm.Create_cart(dims=dims, periods=[False, False, False])
    return mpi_cart_comm

@pytest.fixture
def cart_context(mpi_cart_comm):
    comm = ghex.mpi_comm(mpi_cart_comm)
    ctx = ghex.context(comm, True)
    return ctx
