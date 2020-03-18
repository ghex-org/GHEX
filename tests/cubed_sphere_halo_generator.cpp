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

#include <ghex/structured/pattern.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/threads/none/primitives.hpp>
#include <ghex/cubed_sphere/halo_generator.hpp>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#else
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#endif
using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>


TEST(halo_generator, domain)
{
    using namespace gridtools::ghex::cubed_sphere;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    //domain_descriptor domain(20, 1, 0, std::array<int,3>{10,10,10}, std::array<int,3>{19,19,19});
    //domain_descriptor domain(20, 1, 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,9});
    domain_descriptor domain(10, context.rank(), 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,5});
    //domain_descriptor domain(10, context.rank()==0?1:3, 0, std::array<int,3>{0,0,0}, std::array<int,3>{9,9,9});
    halo_generator halo_gen(2);
    
    domain_descriptor domain0 (10, context.rank(), 0, std::array<int,3>{0,0,0}, std::array<int,3>{4,4,5});
    domain_descriptor domain1 (10, context.rank(), 1, std::array<int,3>{5,0,0}, std::array<int,3>{9,4,5});
    domain_descriptor domain2 (10, context.rank(), 2, std::array<int,3>{0,5,0}, std::array<int,3>{4,9,5});
    domain_descriptor domain3 (10, context.rank(), 3, std::array<int,3>{5,5,0}, std::array<int,3>{9,9,5});
    
    //std::vector<domain_descriptor> local_domains{ domain };
    std::vector<domain_descriptor> local_domains{ domain0, domain1, domain2, domain3 };
    auto pattern1 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);

    MPI_Barrier(context.mpi_comm());
    for (int r=0; r<context.size(); ++r) {
        if (r==context.rank()) {
            std::cout << "rank " << context.rank() << std::endl;
            for (int pp=0; pp<4; ++pp) {
                std::cout << "--tile = " << context.rank() << ", id = " << pp << std::endl;
            for (const auto& kvp : pattern1[pp].send_halos()) {
                const auto& ext_dom_id = kvp.first;
                const auto& idx_cont = kvp.second;
                std::cout << "  sending to tile " << ext_dom_id.id.tile << ", id " << ext_dom_id.id.id << " on rank " 
                << ext_dom_id.mpi_rank << " with tag " << ext_dom_id.tag << std::endl;
                for (const auto& isp : idx_cont) {
                    std::cout 
                        << "    iteration space \n"
                        << "      global: " << "                  " << isp.global().last() << "\n"
                        << "              " << isp.global().first() << "\n"
                        << "      local:  " << "                  " << isp.local().last() << "\n"
                        << "              " << isp.local().first() << std::endl;
                }
            }
            }
        }
        MPI_Barrier(context.mpi_comm());
    }
    MPI_Barrier(context.mpi_comm());
    for (int r=0; r<context.size(); ++r) {
        if (r==context.rank()) {
            std::cout << "rank " << context.rank() << std::endl;
            for (int pp=0; pp<4; ++pp) {
                std::cout << "--tile = " << context.rank() << ", id = " << pp << std::endl;
            for (const auto& kvp : pattern1[pp].recv_halos()) {
                const auto& ext_dom_id = kvp.first;
                const auto& idx_cont = kvp.second;
                std::cout << "  receiving from tile " << ext_dom_id.id.tile << ", id " << ext_dom_id.id.id << " on rank " 
                << ext_dom_id.mpi_rank << " with tag " << ext_dom_id.tag << std::endl;
                for (const auto& isp : idx_cont) {
                    std::cout 
                        << "    iteration space \n"
                        << "      global: " << "                  " << isp.global().last() << "\n"
                        << "              " << isp.global().first() << "\n"
                        << "      local:  " << "                  " << isp.local().last() << "\n"
                        << "              " << isp.local().first() << std::endl;
                }
            }
            }
        }
        MPI_Barrier(context.mpi_comm());
    }
}
