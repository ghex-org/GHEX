import pytest

from mpi4py import MPI

import ghex

@pytest.fixture
def mpi_cart_comm():
    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), [0, 0, 0])
    mpi_cart_comm = mpi_comm.Create_cart(dims=dims, periods=[False, False, False])
    return mpi_cart_comm

@pytest.fixture
def ghex_cart_context(mpi_cart_comm):
    return ghex.tl.Context(mpi_cart_comm)