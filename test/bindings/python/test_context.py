#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
import pytest

from fixtures.context import *


@pytest.mark.mpi_skip
def test_module(capsys):
    with capsys.disabled():
        print(ghex.__version__)
        print(ghex.__config__)


@pytest.mark.mpi_skip
def test_mpi_comm():
    with pytest.raises(TypeError, match=r"must be `mpi4py.MPI.Comm`"):
        comm = ghex.mpi_comm("invalid")


@pytest.mark.mpi_skip
def test_context_mpi4py():
    comm = ghex.mpi_comm(MPI.COMM_WORLD)
    ctx = ghex.context(comm, True)
    assert ctx.size() == 1
    assert ctx.rank() == 0


@pytest.mark.mpi
def test_context(context):
    ctx = context
    assert ctx.size() == 4
