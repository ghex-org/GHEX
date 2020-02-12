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
#include <ghex/threads/atomic/primitives.hpp>
#include <gridtools/storage/storage_facility.hpp>
#include <gtest/gtest.h>

using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::atomic::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

TEST(data_store, make)
{
    const int Nx0 = 10;
    const int Ny0 = 12;
    const int Nz0 = 20;

    const std::array<bool, 3> periodicity{true, true, false};

    using halo_t = gridtools::halo<3,3,0>;

    const int Nx = Nx0+2*halo_t::at<0>();
    const int Ny = Ny0+2*halo_t::at<1>();
    const int Nz = Nz0+2*halo_t::at<2>();

    int np;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm CartComm;
    std::array<int, 3> dimensions{0, 0, 1};
    int period[3] = {1, 1, 1};
    MPI_Dims_create(np, 3, &dimensions[0]);
    MPI_Cart_create(MPI_COMM_WORLD, 3, &dimensions[0], period, false, &CartComm);
    const std::array<int, 3>  extents{Nx0,Ny0,Nz0};

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, CartComm);
    auto& context = *context_ptr;

    auto grid     = gridtools::ghex::make_gt_processor_grid(context, extents, periodicity); 
    auto pattern1 = gridtools::ghex::make_gt_pattern(grid, std::array<int,6>{1,1,1,1,0,0});
    auto co       = gridtools::ghex::make_communication_object<decltype(pattern1)>(context.get_communicator(context.get_token()));

    using host_backend_t        = gridtools::backend::mc;
    using host_storage_info_t   = gridtools::storage_traits<host_backend_t>::storage_info_t<0, 3, halo_t>;
    using host_data_store_t     = gridtools::storage_traits<host_backend_t>::data_store_t<double, host_storage_info_t>;
#ifdef __CUDACC__
    //using target_backend_t      = gridtools::backend::cuda;
#else
    //using target_backend_t      = gridtools::backend::mc;
#endif
    //using target_storage_info_t = gridtools::storage_traits<target_backend_t>::select_storage_info<0, 3, halo_t>;
    //using target_data_store_t   = gridtools::storage_traits<host_backend_t>::data_store_t<double, target_storage_info_t>;

    host_storage_info_t   host_info(Nx, Ny, Nz);
    host_data_store_t     host_data_store(host_info, -1., "field");
    //target_storage_info_t target_info(Nx, Ny, Nz);
    //target_data_store_t   target_data_store(target_info, -1., "field");

    auto host_ghex_field   = gridtools::ghex::wrap_gt_field(grid, host_data_store);

    auto host_view = gridtools::make_host_view(host_data_store);


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
        const bool passed_this = (host_ghex_field(x-(int)halo_t::at<0>(),y-(int)halo_t::at<1>(),z-(int)halo_t::at<2>()) == i++);
        passed = passed && passed_this;
    }

    //auto target_view = gridtools::make_target_view(host_data_store);

    EXPECT_TRUE(passed);

    auto h = co.exchange( pattern1(host_ghex_field) );
    h.wait();

    MPI_Comm_free(&CartComm);
}

