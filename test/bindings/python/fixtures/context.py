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

import ghex
from ghex.context import make_context


mpi4py.rc.initialize = True
mpi4py.rc.finalize = True
mpi4py.rc.threads = True
mpi4py.rc.thread_level = "multiple"


@pytest.fixture(params=[True, False], ids=["thread_safe", "not_thread_safe"])
def thread_safe(request):
    return request.param


@pytest.fixture
def context(thread_safe):
    if ghex.__config__["transport"] == "NCCL" and thread_safe:
        pytest.skip("NCCL not supported with thread_safe = true")
    ctx = make_context(MPI.COMM_WORLD, thread_safe)
    yield ctx
    # Explicit cleanup to ensure UCX/MPI resources are released.
    # Necessary to prevent state accumulation between tests that can cause
    # subsequent parallel tests to hang (especially with UCX backend).
    del ctx
    import gc

    gc.collect()


@pytest.fixture
def mpi_cart_comm():
    mpi_comm = MPI.COMM_WORLD
    dims = MPI.Compute_dims(mpi_comm.Get_size(), [0, 0, 0])
    mpi_cart_comm = mpi_comm.Create_cart(dims=dims, periods=[False, False, False])
    yield mpi_cart_comm
    # Explicitly free the communicator to clean up UCX/MPI state.
    # Without this, subsequent parallel tests may hang due to leftover state
    # from previous tests (particularly when mixing structured and unstructured tests).
    mpi_cart_comm.Free()


@pytest.fixture
def cart_context(mpi_cart_comm, thread_safe):
    if ghex.__config__["transport"] == "NCCL" and thread_safe:
        pytest.skip("NCCL not supported with thread_safe = true")
    ctx = make_context(mpi_cart_comm, thread_safe)
    yield ctx
    # Explicit cleanup to ensure UCX/MPI resources are released.
    # Necessary to prevent state accumulation between tests that can cause
    # subsequent parallel tests to hang (especially with UCX backend).
    del ctx
    import gc

    gc.collect()
