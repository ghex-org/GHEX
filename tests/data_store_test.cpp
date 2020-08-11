/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */

#include <ghex/communication_object_2.hpp>
#include <ghex/glue/gridtools/gt_glue.hpp>
#include <ghex/transport_layer/mpi/context.hpp>
#include <gridtools/storage/builder.hpp>
#include <gridtools/storage/cpu_ifirst.hpp>
#include <gtest/gtest.h>

using transport = gridtools::ghex::tl::mpi_tag;
using context_type = gridtools::ghex::tl::context<transport>;

TEST(data_store, make)
{
    const int Nx0 = 10;
    const int Ny0 = 12;
    const int Nz0 = 20;

    const std::array<bool, 3> periodicity{true, true, false};

    const std::array<int, 3> halo{3, 3, 0};

    const int Nx = Nx0+2*halo[0];
    const int Ny = Ny0+2*halo[1];
    const int Nz = Nz0+2*halo[2];

    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm CartComm;
    std::array<int, 3> dimensions{0, 0, 1};
    int period[3] = {1, 1, 1};
    MPI_Dims_create(np, 3, &dimensions[0]);
    MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0], period, false, &CartComm);
    const std::array<int, 3>  extents{Nx0,Ny0,Nz0};

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(CartComm);
    auto& context = *context_ptr;

    auto grid     = gridtools::ghex::make_gt_processor_grid(context, extents, periodicity);
    auto pattern1 = gridtools::ghex::make_gt_pattern(grid, std::array<int,6>{1,1,1,1,0,0});
    auto co       = gridtools::ghex::make_communication_object<decltype(pattern1)>(context.get_communicator());

    auto host_data_store = gridtools::storage::builder<gridtools::storage::cpu_ifirst>
        .type<double>()
        .halos(halo[0], halo[1], halo[2])
        .dimensions(Nx, Ny, Nz)
        .value(-1.0)
        .name("field")
        .build();

    auto host_ghex_field = gridtools::ghex::wrap_gt_field<gridtools::ghex::cpu>(grid, host_data_store, halo);

    auto host_view = host_data_store->host_view();


    unsigned long int i = 0;
    for (int z=0; z<Nz; ++z)
    for (int y=0; y<Ny; ++y)
    for (int x=0; x<Nx; ++x)
    {
        host_view(x,y,z) = i++;
    }

    bool passed = true;
    i=0;
    for (int z=0; z<Nz; ++z)
    for (int y=0; y<Ny; ++y)
    for (int x=0; x<Nx; ++x)
    {
        const bool passed_this = (host_ghex_field(x-halo[0],y-halo[1],z-halo[2]) == i++);
        passed = passed && passed_this;
    }

    EXPECT_TRUE(passed);

    auto h = co.exchange( pattern1(host_ghex_field) );
    h.wait();

    MPI_Comm_free(&CartComm);
}
