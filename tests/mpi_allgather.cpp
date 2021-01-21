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
    gridtools::ghex::tl::mpi::rank_topology t{mpi_comm};
    gridtools::ghex::tl::mpi::setup_communicator comm{t};

    std::vector<T> values = comm.all_gather( static_cast<T>(comm.address()) );
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

template<typename T, typename Comm>
bool check_values(const std::vector<std::vector<T>>& values, Comm comm)
{
    bool passed = true;
    if (values.size() != (unsigned)comm.size()) passed =  false;;
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
    return passed;
}

TEST(all_gather, all_gather_vector)
{
    using T = double;
    gridtools::ghex::tl::mpi::communicator_base mpi_comm;
    gridtools::ghex::tl::mpi::rank_topology t{mpi_comm};
    gridtools::ghex::tl::mpi::setup_communicator comm{t};

    int my_num_values = (comm.address()+1)*2;
    std::vector<T> my_values(my_num_values);
    for (int i=0; i<my_num_values; ++i)
        my_values[i] = (comm.address()+1)*1000 + i;

    // simple allgather
    {
        auto num_values = comm.all_gather(my_num_values);
        auto values = comm.all_gather(my_values, num_values);
        EXPECT_TRUE(check_values(values, mpi_comm));
    }

    // allgather with skeleton
    {
        auto sizes = comm.all_gather_sizes(my_num_values);
        auto skeleton = comm.all_gather_skeleton<T>(sizes);
        auto values = comm.all_gather(my_values, skeleton, sizes);
        EXPECT_TRUE(check_values(values, mpi_comm));
    }

    // allgather with implicit skeleton
    {
        auto sizes = comm.all_gather_sizes(my_num_values);
        auto values = comm.all_gather(my_values, sizes);
        EXPECT_TRUE(check_values(values, mpi_comm));
    }
}

