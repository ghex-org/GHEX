/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <mpi.h>

struct mpi_test_fixture : public ::testing::Test
{
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override
    {
        world = MPI_COMM_WORLD;
        MPI_Comm_rank(world, &world_rank);
        MPI_Comm_size(world, &world_size);
        int mpi_thread_safety;
        MPI_Query_thread(&mpi_thread_safety);
        thread_safe = (mpi_thread_safety == MPI_THREAD_MULTIPLE);
    }

    //void TearDown() {}

  protected:
    MPI_Comm world;
    int      world_rank;
    int      world_size;
    bool     thread_safe;
};
