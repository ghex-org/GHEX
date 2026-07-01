#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
from mpi4py import MPI
import pytest

import ghex
from ghex.context import make_context


@pytest.mark.mpi_skip
def test_module(capsys):
    with capsys.disabled():
        print(ghex.__version__)
        print(ghex.__config__)


@pytest.mark.mpi_skip
def test_mpi_comm():
    with pytest.raises(TypeError, match=r"must be `mpi4py.MPI.Comm`"):
        comm = ghex.mpi_comm("invalid")


@pytest.mark.parametrize("thread_safe", [True, False], ids=["thread_safe", "not_thread_safe"])
@pytest.mark.mpi_skip
def test_context_mpi4py(thread_safe):
    try:
        ctx = make_context(MPI.COMM_WORLD, thread_safe)
        assert ctx.size() == 1
        assert ctx.rank() == 0
    except RuntimeError as e:
        if ghex.__config__["transport"] == "NCCL" and thread_safe:
            assert str(e) == "NCCL not supported with thread_safe = true"
        else:
            raise


@pytest.mark.mpi
def test_context(context):
    ctx = context
    assert ctx.size() == 4
