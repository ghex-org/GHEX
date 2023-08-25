/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ghex/mpi/communicator.hpp>
#include <gtest/gtest.h>
#include "./mpi_runner/mpi_test_fixture.hpp"

TEST_F(mpi_test_fixture, all_gather_fixed)
{
    using namespace ghex;
    using T = double;

    context           ctxt(world, thread_safe);
    mpi::communicator comm(ctxt);

    std::vector<T> values;
    {
        auto f = comm.all_gather(static_cast<T>(comm.rank()));
        values = f.get();
    }
    bool passed = true;
    int  i = 0;
    for (const auto& v : values)
    {
        if (v != static_cast<T>(i)) passed = false;
        ++i;
    }
    EXPECT_TRUE(passed);
}

TEST_F(mpi_test_fixture, all_gather)
{
    using namespace ghex;
    using T = double;

    context           ctxt(world, thread_safe);
    mpi::communicator comm(ctxt);

    int            my_num_values = (comm.rank() + 1) * 2;
    std::vector<T> my_values(my_num_values);
    for (int i = 0; i < my_num_values; ++i) my_values[i] = (comm.rank() + 1) * 1000 + i;

    auto num_values = comm.all_gather(my_num_values).get();
    auto values = comm.all_gather(my_values, num_values).get();

    bool passed = true;
    if (values.size() != (unsigned)comm.size()) passed = false;
    ;
    int i = 0;
    for (const auto& vec : values)
    {
        if (vec.size() != (unsigned)((i + 1) * 2)) passed = false;
        int j = 0;
        for (const auto& v : vec)
        {
            if (v != static_cast<T>((i + 1) * 1000 + j)) passed = false;
            ++j;
        }
        ++i;
    }
    EXPECT_TRUE(passed);
}
