#
# ghex-org
#
# Copyright (c) 2014-2023, ETH Zurich
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
#
import numpy as np
import pytest

from mpi4py import MPI

try:
    import cupy as cp
except ImportError:
    cp = None

import ghex
from ghex.context import make_context
from ghex.util import Architecture
from ghex.structured.cartesian_sets import IndexSpace
from ghex.structured.regular import (
    make_communication_object,
    DomainDescriptor,
    make_field_descriptor,
    HaloGenerator,
    make_pattern,
)


# Global domain size. The first `ndim` entries are used, so the grid is a genuine
# 1d/2d/3d domain. Every used size is larger than one (the degenerate Nz == 1 case
# is intentionally not tested).
sizes = (48, 24, 16)

# Halo widths per dimension.
halos_per_dim = ((2, 1), (1, 2), (1, 1))


@pytest.mark.mpi
@pytest.mark.parametrize("gpu_and_stream", (
    (False, None),
    (True, None),
    (True, {"null": True}),
    (True, {"non_blocking": True}),
))
@pytest.mark.parametrize("periodic", [True, False])
@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_pattern(capsys, ndim, periodic, gpu_and_stream):
    gpu, stream_args = gpu_and_stream
    if gpu:
        if cp is None:
            pytest.skip("`CuPy` is not installed.")
        if not cp.is_available():
            pytest.skip("`CuPy` is installed but no GPU could be found.")
        if not ghex.__config__["gpu"]:
            pytest.skip("`GHEX` was not compiled with GPU support.")
        xp = cp
        arch = Architecture.GPU
    else:
        xp = np
        arch = Architecture.CPU
    # `stream_args is None` selects a plain exchange, with the cupy arrays living
    # on the cupy default stream; otherwise a scheduled exchange is run on the
    # requested stream (the null/default stream or a non-blocking one).
    cuda_stream = cp.cuda.Stream(**stream_args) if stream_args else None

    mpi_comm = MPI.COMM_WORLD

    # decompose all `ndim` dimensions over the ranks
    dims = MPI.Compute_dims(mpi_comm.Get_size(), [0] * ndim)

    halos = tuple(halos_per_dim[d] for d in range(ndim))

    # toggle the periodicity of the first dimension; the others are periodic
    periodicity = tuple(periodic if d == 0 else True for d in range(ndim))

    mpi_cart_comm = mpi_comm.Create_cart(dims=dims, periods=list(periodicity))

    ctx = make_context(mpi_cart_comm, True)

    p_coord = tuple(mpi_cart_comm.Get_coords(mpi_cart_comm.Get_rank()))
    global_grid = IndexSpace.from_sizes(*sizes[:ndim])
    sub_grids = global_grid.decompose(mpi_cart_comm.dims)
    owned_indices = sub_grids[p_coord].subset["definition"]  # sub-grid in global coordinates
    sub_grid = IndexSpace(
        {
            "definition": owned_indices,
            "halo": owned_indices.extend(*halos).without(owned_indices),
        }
    )

    memory_local_grid = sub_grid.translate(
        *(-origin_l for origin_l in sub_grid.bounds[(0,) * ndim])
    )

    domain_desc = DomainDescriptor(ctx.rank(), owned_indices)
    halo_gen = HaloGenerator(global_grid.subset["definition"], halos, periodicity)

    pattern = make_pattern(ctx, halo_gen, [domain_desc])

    with capsys.disabled():
        print("python side: making co")
        print(pattern.grid_type)
        print(pattern.domain_id_type)

    co = make_communication_object(ctx)

    def make_field():
        field_1 = xp.zeros(memory_local_grid.bounds.shape, dtype=np.float64, order="F")
        gfield_1 = make_field_descriptor(
            domain_desc,
            field_1,
            memory_local_grid.subset["definition"][(0,) * ndim],
            memory_local_grid.bounds.shape,
            arch=arch,
        )
        return field_1, gfield_1

    def exchange(buffer_infos):
        if cuda_stream is None:
            if gpu:
                cp.cuda.Device().synchronize()
            res = co.exchange(buffer_infos)
            res.wait()
        else:
            # The fields were initialized on the cupy default stream. Unless we are
            # scheduling on that same (null) stream, make `cuda_stream` wait for
            # them so they are not packed prematurely.
            if not stream_args.get("null"):
                cuda_stream.wait_event(cp.cuda.get_current_stream().record())
            res = co.schedule_exchange(cuda_stream, buffer_infos)
            assert not co.has_scheduled_exchange()
            res.schedule_wait(cuda_stream)
            assert co.has_scheduled_exchange()
            res.wait()
            assert not co.has_scheduled_exchange()

    # one field per dimension, each storing the owner's coordinate in that dimension
    fields = []
    gfields = []
    for _ in range(ndim):
        field, gfield = make_field()
        fields.append(field)
        gfields.append(gfield)
    for p_dim, p_coord_l in enumerate(p_coord):
        fields[p_dim][...] = p_coord_l

    exchange([pattern(gfield) for gfield in gfields])

    rank_field, grank_field = make_field()
    rank_field[...] = ctx.rank()
    exchange(
        [pattern(grank_field)]
    )  # arch, dtype. exchange of fields living on cpu+gpu possible

    # copy the results back to the host for checking
    if gpu:
        fields = [cp.asnumpy(field) for field in fields]
        rank_field = cp.asnumpy(rank_field)

    with capsys.disabled():
        print("post_ex:")
        print(rank_field)

    last = global_grid.subset["definition"][(-1,) * ndim]
    for m_idx, local_idx in zip(memory_local_grid.bounds, sub_grid.bounds):
        value_owner_coord = tuple(int(fields[dim][m_idx]) for dim in range(ndim))
        value_owner_rank = mpi_cart_comm.Get_cart_rank(value_owner_coord)
        if all(l >= 0 and l <= last[d] for d, l in enumerate(local_idx)):
            assert local_idx in sub_grids[value_owner_coord].subset["definition"]

            assert rank_field[m_idx] == value_owner_rank
