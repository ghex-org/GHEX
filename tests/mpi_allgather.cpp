//
// GridTools
//
// Copyright (c) 2014-2020, ETH Zurich
// All rights reserved.
//
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
//

#include <ghex/transport_layer/mpi/setup.hpp>
#include <gtest/gtest.h>

TEST(all_gather, all_gather_fixed)
{
    using T = double;
    gridtools::ghex::tl::mpi::communicator_base mpi_comm;
    gridtools::ghex::tl::mpi::setup_communicator comm{mpi_comm};

    std::vector<T> values;
    {
        auto f = comm.all_gather( static_cast<T>(comm.address()) );
        values = f.get();
    }
    bool passed = true;
    int i = 0;
    for (const auto& v : values)
    {
        if (v != static_cast<T>(i))
            passed = false;
        ++i;
    }
    EXPECT_TRUE(passed);
}

TEST(all_gather, all_gather_vector)
{
    using T = double;
    gridtools::ghex::tl::mpi::communicator_base mpi_comm;
    gridtools::ghex::tl::mpi::setup_communicator comm{mpi_comm};

    int my_num_values = (comm.address()+1)*2;
    std::vector<T> my_values(my_num_values);
    for (int i=0; i<my_num_values; ++i)
        my_values[i] = (comm.address()+1)*1000 + i;

    auto num_values = comm.all_gather(my_num_values).get();
    auto values = comm.all_gather(my_values, num_values).get();

    bool passed = true;
    if (values.size() != (unsigned)mpi_comm.size()) passed =  false;;
    int i = 0;
    for (const auto& vec : values)
    {
        if (vec.size() != (unsigned)((i+1)*2)) passed = false;
        int j = 0;
        for (const auto& v : vec)
        {
            if (v != static_cast<T>((i+1)*1000+j)) passed = false;
            ++j;
        }
        ++i;
    }
    EXPECT_TRUE(passed);
}

