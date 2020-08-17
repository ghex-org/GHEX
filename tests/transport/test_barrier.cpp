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
#include <iomanip>
#include <thread>

#include <ghex/transport_layer/util/barrier.hpp>
#include <ghex/common/timer.hpp>
#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif


TEST(transport, rank_barrier) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;

    gridtools::ghex::tl::barrier_t barrier;

    auto comm = context.get_communicator();
    int rank = context.rank();
    gridtools::ghex::timer timer;

    timer.tic();
    for(int i=0; i<100; i++)  {
        barrier.rank_barrier(comm);
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

TEST(transport, in_barrier) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;

    size_t n_threads = 8;
    gridtools::ghex::tl::barrier_t barrier{n_threads};

    auto work =
        [&]()
        {
            auto comm = context.get_communicator();
            int rank = context.rank();
            gridtools::ghex::timer timer;

            // auto token = barrier.get_token();
            // barrier.wait_registration();

            timer.tic();
            for(int i=0; i<100; i++)  {
                comm.progress();
                barrier.in_node();
            }
            const auto t = timer.stoc();
            if(rank==0)
                {
                    std::cout << "time:       " << t/1000000 << "s\n";
                }
        };

    std::vector<std::thread> ths;
    for (size_t i = 0; i < n_threads; ++i) {
        ths.push_back(std::thread{work});
    }
    for (size_t i = 0; i < n_threads; ++i) {
        ths[i].join();
    }

}

TEST(transport, full_barrier) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;

    size_t n_threads = 8;
    gridtools::ghex::tl::barrier_t barrier{n_threads};

    auto work =
        [&]()
        {
            auto comm = context.get_communicator();
            int rank = context.rank();
            gridtools::ghex::timer timer;

            timer.tic();
            for(int i=0; i<100; i++)  {
                barrier(comm);
            }
            const auto t = timer.stoc();
            if(rank==0)
                {
                    std::cout << "time:       " << t/1000000 << "s\n";
                }
        };

    std::vector<std::thread> ths;
    for (size_t i = 0; i < n_threads; ++i) {
        ths.push_back(std::thread{work});
    }
    for (size_t i = 0; i < n_threads; ++i) {
        ths[i].join();
    }

}
