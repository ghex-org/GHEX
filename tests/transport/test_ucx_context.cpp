/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#include <iostream>
#include <ghex/transport_layer/ucx3/address_db_mpi.hpp>
#include <ghex/transport_layer/ucx/context.hpp>
#include <ghex/threads/atomic/primitives.hpp>

#include <gtest/gtest.h>

#include <thread>
#include <vector>

namespace ghex = gridtools::ghex;

using db_type      = ghex::tl::ucx::address_db_mpi;
using transport    = gridtools::ghex::tl::ucx_tag;
using threading    = gridtools::ghex::threads::atomic::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;


TEST(transport_layer, ucx_context)
{
    int num_threads = 4;

    context_type context{num_threads, MPI_COMM_WORLD, db_type{MPI_COMM_WORLD} };

    auto func = [&context]()
    {
        auto token = context.get_token();

        auto comm = context.get_communicator(token);

        std::vector<int> payload{1,2,3,4};

        comm.send(payload, 0, token.id()).wait();

        //const auto& ep = comm.m_send_worker->connect(0);

        context.barrier(token);
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i=0; i<num_threads; ++i)
        threads.push_back(std::thread(func));

    for (auto& t : threads)
        t.join();
}
