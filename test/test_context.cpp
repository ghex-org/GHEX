/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ghex/context.hpp>
#include <ghex/barrier.hpp>
#include <gtest/gtest.h>
#include "./mpi_runner/mpi_test_fixture.hpp"
#include <iostream>
#include <vector>
#include <thread>

TEST_F(mpi_test_fixture, context)
{
    using namespace ghex;

    context ctxt(world, thread_safe);
}

#if OOMPH_ENABLE_BARRIER
TEST_F(mpi_test_fixture, barrier)
{
    using namespace ghex;

    context ctxt(world, thread_safe);

    if (thread_safe)
    {
        barrier b(ctxt, 1);
        b.rank_barrier();
    }
    else
    {
        barrier b(ctxt, 4);

        auto use_barrier = [&]() { b(); };

        auto use_thread_barrier = [&]() { b.thread_barrier(); };

        std::vector<std::thread> threads;
        for (int i = 0; i < 4; ++i) threads.push_back(std::thread{use_thread_barrier});
        for (int i = 0; i < 4; ++i) threads[i].join();
        threads.clear();
        for (int i = 0; i < 4; ++i) threads.push_back(std::thread{use_barrier});
        for (int i = 0; i < 4; ++i) threads[i].join();
    }
}
#endif
