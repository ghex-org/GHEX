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

#include <ghex/util/decomposition.hpp>
#include <gtest/gtest.h>


void test_distribution_1(int X)
{
    using namespace gridtools::ghex;

    dims_map<1> dist_a({X}, false);
    for (int x=0; x<X; ++x)
        EXPECT_EQ(dist_a(x)[0], x);

    dims_map<1> dist_b({X}, true);
    for (int x=0; x<X; ++x)
        EXPECT_EQ(dist_b(x)[0], x);
}

void test_distribution_2(int X, int Y)
{
    using namespace gridtools::ghex;

    dims_map<2> dist_a({X,Y}, false);
    int i=0;
    for (int y=0; y<Y; ++y)
        for (int x=0; x<X; ++x)
        {
            EXPECT_EQ(dist_a(i)[0], x);
            EXPECT_EQ(dist_a(i)[1], y);
            ++i;
        }

    dims_map<2> dist_b({X,Y}, true);
    i=0;
    for (int x=0; x<X; ++x)
        for (int y=0; y<Y; ++y)
        {
            EXPECT_EQ(dist_b(i)[0], x);
            EXPECT_EQ(dist_b(i)[1], y);
            ++i;
        }
}

void test_distribution_3(int X, int Y, int Z)
{
    using namespace gridtools::ghex;

    dims_map<3> dist_a({X,Y,Z}, false);
    int i=0;
    for (int z=0; z<Z; ++z)
    for (int y=0; y<Y; ++y)
        for (int x=0; x<X; ++x)
        {
            EXPECT_EQ(dist_a(i)[0], x);
            EXPECT_EQ(dist_a(i)[1], y);
            EXPECT_EQ(dist_a(i)[2], z);
            ++i;
        }

    dims_map<3> dist_b({X,Y,Z}, true);
    i=0;
    for (int x=0; x<X; ++x)
    for (int y=0; y<Y; ++y)
        for (int z=0; z<Z; ++z)
        {
            EXPECT_EQ(dist_b(i)[0], x);
            EXPECT_EQ(dist_b(i)[1], y);
            EXPECT_EQ(dist_b(i)[2], z);
            ++i;
        }
}

TEST(decompostion, distribution)
{
    for (int i=0; i<5; ++i)
        test_distribution_1(i);

    for (int i=0; i<5; ++i)
    for (int j=0; j<5; ++j)
        test_distribution_2(i, j);

    for (int i=0; i<5; ++i)
    for (int j=0; j<5; ++j)
    for (int k=0; k<5; ++k)
        test_distribution_3(i, j, k);
}

