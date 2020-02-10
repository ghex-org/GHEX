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
#include <iostream>
#include <ghex/transport_layer/ucx/address_db_mpi.hpp>
#include <ghex/transport_layer/ucx/context.hpp>
#include <ghex/threads/atomic/primitives.hpp>

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace ghex = gridtools::ghex;

using db_type      = ghex::tl::ucx::address_db_mpi;
using transport    = ghex::tl::ucx_tag;
using threading    = ghex::threads::atomic::primitives;
using context_type = ghex::tl::context<transport, threading>;


TEST(transport_layer, ucx_context)
{
    int num_threads = 4;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto func = [&context]()
    {
        auto token = context.get_token();

        auto comm = context.get_communicator(token);

        std::vector<int> payload{1,2,3,4};

        if (comm.rank() == 0)
        {
            for (int i=1; i<comm.size(); ++i)
            {
                comm.recv(payload, i, token.id()).wait();
                EXPECT_EQ(payload[0], token.id());
                EXPECT_EQ(payload[1], i);
            }
        }
        else
        {
            payload[0] = token.id();
            payload[1] = comm.rank();
            comm.send(payload, 0, token.id()).wait();
        }

        comm.barrier();
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i=0; i<num_threads; ++i)
        threads.push_back(std::thread(func));

    for (auto& t : threads)
        t.join();
}
