/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <random>
#include <ghex/util/decomposition.hpp>
#include <gtest/gtest.h>

// test 1D distribution
void
test_distribution_1(int X)
{
    using namespace ghex;

    dims_map<1> dist_a({X}, false);
    for (int x = 0; x < X; ++x) EXPECT_EQ(dist_a(x)[0], x);

    dims_map<1> dist_b({X}, true);
    for (int x = 0; x < X; ++x) EXPECT_EQ(dist_b(x)[0], x);
}

// test 2D distribution
void
test_distribution_2(int X, int Y)
{
    using namespace ghex;

    dims_map<2> dist_a({X, Y}, false);
    int         i = 0;
    for (int y = 0; y < Y; ++y)
        for (int x = 0; x < X; ++x)
        {
            EXPECT_EQ(dist_a(i)[0], x);
            EXPECT_EQ(dist_a(i)[1], y);
            ++i;
        }

    dims_map<2> dist_b({X, Y}, true);
    i = 0;
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y)
        {
            EXPECT_EQ(dist_b(i)[0], x);
            EXPECT_EQ(dist_b(i)[1], y);
            ++i;
        }
}

// test 3D distribution
void
test_distribution_3(int X, int Y, int Z)
{
    using namespace ghex;

    dims_map<3> dist_a({X, Y, Z}, false);
    int         i = 0;
    for (int z = 0; z < Z; ++z)
        for (int y = 0; y < Y; ++y)
            for (int x = 0; x < X; ++x)
            {
                EXPECT_EQ(dist_a(i)[0], x);
                EXPECT_EQ(dist_a(i)[1], y);
                EXPECT_EQ(dist_a(i)[2], z);
                ++i;
            }

    dims_map<3> dist_b({X, Y, Z}, true);
    i = 0;
    for (int x = 0; x < X; ++x)
        for (int y = 0; y < Y; ++y)
            for (int z = 0; z < Z; ++z)
            {
                EXPECT_EQ(dist_b(i)[0], x);
                EXPECT_EQ(dist_b(i)[1], y);
                EXPECT_EQ(dist_b(i)[2], z);
                ++i;
            }
}

TEST(decompostion, distribution)
{
    for (int i = 0; i < 5; ++i) test_distribution_1(i);

    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j) test_distribution_2(i, j);

    for (int i = 0; i < 5; ++i)
        for (int j = 0; j < 5; ++j)
            for (int k = 0; k < 5; ++k) test_distribution_3(i, j, k);
}

// test 3D hierarchical decomposition with 4 levels
void
test_decomposition_3_4(int node_X, int node_Y, int node_Z, int numa_X, int numa_Y, int numa_Z,
    int rank_X, int rank_Y, int rank_Z, int thread_X, int thread_Y, int thread_Z)
{
    using namespace ghex;
    using decomp_t = hierarchical_decomposition<3>;

    decomp_t decomp({node_X, node_Y, node_Z}, {numa_X, numa_Y, numa_Z}, {rank_X, rank_Y, rank_Z},
        {thread_X, thread_Y, thread_Z});

    // check total number of domains
    EXPECT_EQ(decomp.size(), node_X * node_Y * node_Z * numa_X * numa_Y * numa_Z * rank_X * rank_Y *
                                 rank_Z * thread_X * thread_Y * thread_Z);

    // check number of nodes
    EXPECT_EQ(decomp.nodes(), node_X * node_Y * node_Z);

    // check number of numa nodes per node
    EXPECT_EQ(decomp.numas_per_node(), numa_X * numa_Y * numa_Z);

    // check number of ranks per numa node
    EXPECT_EQ(decomp.ranks_per_numa(), rank_X * rank_Y * rank_Z);

    // check number of threads per rank
    EXPECT_EQ(decomp.threads_per_rank(), thread_X * thread_Y * thread_Z);

    // loop over decomposition
    int idx = 0;
    int node_idx = 0;
    for (int node_z = 0; node_z < node_Z; ++node_z)
        for (int node_y = 0; node_y < node_Y; ++node_y)
            for (int node_x = 0; node_x < node_X; ++node_x)
            {
                int node_res = 0;
                int numa_idx = 0;
                for (int numa_z = 0; numa_z < numa_Z; ++numa_z)
                    for (int numa_y = 0; numa_y < numa_Y; ++numa_y)
                        for (int numa_x = 0; numa_x < numa_X; ++numa_x)
                        {
                            int numa_res = 0;
                            int rank_idx = 0;
                            for (int rank_z = 0; rank_z < rank_Z; ++rank_z)
                                for (int rank_y = 0; rank_y < rank_Y; ++rank_y)
                                    for (int rank_x = 0; rank_x < rank_X; ++rank_x)
                                    {
                                        int thread_idx = 0;
                                        for (int thread_z = 0; thread_z < thread_Z; ++thread_z)
                                            for (int thread_y = 0; thread_y < thread_Y; ++thread_y)
                                                for (int thread_x = 0; thread_x < thread_X;
                                                     ++thread_x)
                                                {
                                                    // check level indices
                                                    EXPECT_EQ(decomp.node_index(idx), node_idx);
                                                    EXPECT_EQ(decomp.numa_index(idx), numa_idx);
                                                    EXPECT_EQ(decomp.rank_index(idx), rank_idx);
                                                    EXPECT_EQ(decomp.thread_index(idx), thread_idx);

                                                    // check node resource
                                                    EXPECT_EQ(decomp.node_resource(idx), node_res);

                                                    // check node resource
                                                    EXPECT_EQ(decomp.numa_resource(idx), numa_res);

                                                    // check coordinate
                                                    EXPECT_EQ(decomp(idx)[0],
                                                        thread_x +
                                                            thread_X *
                                                                (rank_x +
                                                                    rank_X *
                                                                        (numa_x +
                                                                            numa_X * (node_x))));
                                                    EXPECT_EQ(decomp(idx)[1],
                                                        thread_y +
                                                            thread_Y *
                                                                (rank_y +
                                                                    rank_Y *
                                                                        (numa_y +
                                                                            numa_Y * (node_y))));
                                                    EXPECT_EQ(decomp(idx)[2],
                                                        thread_z +
                                                            thread_Z *
                                                                (rank_z +
                                                                    rank_Z *
                                                                        (numa_z +
                                                                            numa_Z * (node_z))));

                                                    ++idx;
                                                    ++thread_idx;
                                                    ++node_res;
                                                    ++numa_res;
                                                }
                                        ++rank_idx;
                                    }
                            ++numa_idx;
                        }
                ++node_idx;
            }
}

TEST(decompostion, decomposition)
{
    std::random_device              rd;
    std::mt19937                    gen(rd());
    std::uniform_int_distribution<> distrib(1, 5);

    for (int n = 0; n < 5; ++n)
        test_decomposition_3_4(distrib(gen), distrib(gen), distrib(gen), distrib(gen), distrib(gen),
            distrib(gen), distrib(gen), distrib(gen), distrib(gen), distrib(gen), distrib(gen),
            distrib(gen));
}
