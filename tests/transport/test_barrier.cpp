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
#include <numeric>

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
    for(int i=0; i<20; i++)  {
        barrier.rank_barrier(comm);
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

namespace gridtools {
    namespace ghex {
        namespace tl {
            struct test_barrier {

                gridtools::ghex::tl::barrier_t& br;

                test_barrier(gridtools::ghex::tl::barrier_t& br) : br{br} {}

                template <typename Context>
                void test_in_node1(Context &context) {
                    std::vector<int> innode1_out(br.size());
                    auto work = [&](int id) {
                                    auto comm = context.get_communicator();
                                    innode1_out[id] = br.in_node1(comm)?1:0;
                                };
                    std::vector<std::thread> ths;
                    for (int i = 0; i < br.size(); ++i) {
                        ths.push_back(std::thread{work, i});
                    }
                    for (int i = 0; i < br.size(); ++i) {
                        ths[i].join();
                    }
                    EXPECT_EQ(std::accumulate(innode1_out.begin(), innode1_out.end(), 0), 1);
                }
            };
        }
    }
}


TEST(transport, in_barrier_1) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;

    size_t n_threads = 4;
    gridtools::ghex::tl::barrier_t barrier{n_threads};

    gridtools::ghex::tl::test_barrier test(barrier);
    test.test_in_node1(context);
}

TEST(transport, in_barrier) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;

    size_t n_threads = 4;
    gridtools::ghex::tl::barrier_t barrier{n_threads};

    auto work =
        [&]()
        {
            auto comm = context.get_communicator();
            int rank = context.rank();
            gridtools::ghex::timer timer;

            timer.tic();
            for(int i=0; i<20; i++)  {
                comm.progress();
                barrier.in_node(comm);
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

    size_t n_threads = 4;
    gridtools::ghex::tl::barrier_t barrier{n_threads};

    auto work =
        [&]()
        {
            auto comm = context.get_communicator();
            int rank = context.rank();
            gridtools::ghex::timer timer;

            timer.tic();
            for(int i=0; i<20; i++)  {
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
