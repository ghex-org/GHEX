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

from ghex.context import make_context
from ghex.structured.cartesian_sets import IndexSpace, UnitRange
from ghex.structured.regular import DomainDescriptor, HaloGenerator


# Domain configuration
Nx = 10
Ny = 10
Nz = 2

# halo configurations
haloss = [
    (1, 0, 0),
    (1, 2, 3),
    ((1, 0), (0, 0), (0, 0)),
    ((1, 0), (0, 2), (2, 2)),
]


@pytest.mark.mpi
def test_domain_descriptor(capsys, mpi_cart_comm):
    ctx = make_context(mpi_cart_comm, True)

    coords = mpi_cart_comm.Get_coords(mpi_cart_comm.Get_rank())
    coords2 = mpi_cart_comm.Get_coords(ctx.rank())
    with capsys.disabled():
        print(coords)
        print(coords2)

    assert coords == coords2

    (i, j, k) = coords
    rx = UnitRange(i * Nx, (i + 1) * Nx)
    ry = UnitRange(j * Ny, (j + 1) * Ny)
    rz = UnitRange(k * Nz, (k + 1) * Nz)

    sub_domain_indices = rx * ry * rz

    domain_desc = DomainDescriptor(ctx.rank(), sub_domain_indices)

    assert domain_desc.domain_id() == ctx.rank()
    assert domain_desc.first() == sub_domain_indices[0, 0, 0]
    assert domain_desc.last() == sub_domain_indices[-1, -1, -1]


@pytest.mark.parametrize("halos", haloss)
@pytest.mark.mpi
def test_halo_gen_construction(capsys, mpi_cart_comm, halos):
    with capsys.disabled():
        print(halos)
    dims = mpi_cart_comm.dims
    glob_domain_indices = (
        UnitRange(0, dims[0] * Nx)
        * UnitRange(0, dims[1] * Ny)
        * UnitRange(0, dims[2] * Nz)
    )
    halo_gen = HaloGenerator(glob_domain_indices, halos, (False, False, False))


@pytest.mark.parametrize("halos", haloss)
@pytest.mark.mpi
def test_halo_gen_call(mpi_cart_comm, halos):
    ctx = make_context(mpi_cart_comm, True)

    periodicity = (False, False, False)
    p_coord = tuple(mpi_cart_comm.Get_coords(mpi_cart_comm.Get_rank()))

    # setup grid
    global_grid = IndexSpace.from_sizes(Nx, Ny, Nz)
    sub_grids = global_grid.decompose(mpi_cart_comm.dims)
    sub_grid = sub_grids[p_coord]  # sub-grid in global coordinates
    owned_indices = sub_grid.subset["definition"]
    sub_grid.add_subset("halo", owned_indices.extend(*halos).without(owned_indices))

    # construct halo_generator
    halo_gen = HaloGenerator(global_grid.subset["definition"], halos, periodicity)

    domain_desc = DomainDescriptor(ctx.rank(), owned_indices)

    # test generated halos
    halo_indices = halo_gen(domain_desc)
    assert sub_grid.subset["halo"] == halo_indices.global_
    # assert sub_grid.subset["halo"] == halo_indices.local.translate(...)


@pytest.mark.parametrize("halos", haloss)
@pytest.mark.mpi
def test_domain_descriptor_grid(mpi_cart_comm, halos):
    ctx = make_context(mpi_cart_comm, True)

    p_coord = tuple(mpi_cart_comm.Get_coords(mpi_cart_comm.Get_rank()))

    # setup grid
    global_grid = IndexSpace.from_sizes(Nx, Ny, Nz)
    sub_grids = global_grid.decompose(mpi_cart_comm.dims)
    sub_grid = sub_grids[p_coord]  # sub-grid in global coordinates
    owned_indices = sub_grid.subset["definition"]
    sub_grid.add_subset("halo", owned_indices.extend(*halos).without(owned_indices))

    domain_desc = DomainDescriptor(ctx.rank(), owned_indices)

    assert domain_desc.domain_id() == ctx.rank()
    assert domain_desc.first() == owned_indices.bounds[0, 0, 0]
    assert domain_desc.last() == owned_indices.bounds[-1, -1, -1]
