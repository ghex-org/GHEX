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

#include <ghex/structured/cubed_sphere/transform.hpp>
#include <gtest/gtest.h>
#include <iostream>

template<typename T, unsigned long N>
std::ostream& operator<<(std::ostream& os, const std::array<T, N>& arr) {
    os << "[" << arr[0];
    for (unsigned int n=1; n<N; ++n) os << ", " << arr[n];
    os << "]";
    return os;
}

template<typename T, unsigned long N>
bool operator==(const std::array<T,N>& a, const std::array<T,N>& b) {
    for (unsigned int n=0; n<N; ++n)
        if (a[n] != b[n]) return false;
    return true;
}

// helper function
int reverse_idx_lu(int tile, int ntile) {
    using namespace gridtools::ghex::structured::cubed_sphere;
    int d=0;
    for (auto d0 : tile_lu[ntile]) {
        if (d0==tile)
            break; 
        ++d;
    }
    return d;
}

// check neighbour tiles
void neighbor_test(int tile, int mx, int px, int my, int py) {
    using namespace gridtools::ghex::structured::cubed_sphere;
    EXPECT_EQ(tile_lu[tile][0], mx);
    EXPECT_EQ(tile_lu[tile][1], px);
    EXPECT_EQ(tile_lu[tile][2], my);
    EXPECT_EQ(tile_lu[tile][3], py);
}

// check the transformed halo region
void transform_test(int c, int b, int tile) {
    using namespace gridtools::ghex::structured::cubed_sphere;

    // halo regions along the 4 edges in tile coordinates
    const std::array<std::pair<std::array<int,2>,std::array<int,2>>,4> halo_regions {
        std::make_pair(std::array<int,2>{   -b,    0}, std::array<int,2>{   -1,  c-1}),
        std::make_pair(std::array<int,2>{    c,    0}, std::array<int,2>{c+b-1,  c-1}),
        std::make_pair(std::array<int,2>{    0,   -b}, std::array<int,2>{  c-1,   -1}),
        std::make_pair(std::array<int,2>{    0,    c}, std::array<int,2>{  c-1,c+b-1})};

    // expected halo regions along the 4 edges in respective neighbor tile coordinates
    std::array<std::pair<std::array<int,2>,std::array<int,2>>,4> expected_halo_regions;
    if (tile % 2 == 0) {
        // even tiles
        expected_halo_regions = std::array<std::pair<std::array<int,2>,std::array<int,2>>,4>{
            std::make_pair(std::array<int,2>{  c-1,  c-b}, std::array<int,2>{    0,  c-1}),
            std::make_pair(std::array<int,2>{    0,    0}, std::array<int,2>{  b-1,  c-1}),
            std::make_pair(std::array<int,2>{    0,  c-b}, std::array<int,2>{  c-1,  c-1}),
            std::make_pair(std::array<int,2>{    0,  c-1}, std::array<int,2>{  b-1,    0})};
    }
    else {
        // odd tiles
        expected_halo_regions = std::array<std::pair<std::array<int,2>,std::array<int,2>>,4>{
            std::make_pair(std::array<int,2>{  c-b,    0}, std::array<int,2>{  c-1,  c-1}),
            std::make_pair(std::array<int,2>{  c-1,    0}, std::array<int,2>{    0,  b-1}),
            std::make_pair(std::array<int,2>{  c-b,  c-1}, std::array<int,2>{  c-1,    0}),
            std::make_pair(std::array<int,2>{    0,    0}, std::array<int,2>{  c-1,  b-1})};
    }

    // loop over neighbors
    for (int n=0; n<4; ++n) {
        // get transform to neighbor n
        const auto& t = transform_lu[tile][n];
        // get original halo region
        const auto o_min = halo_regions[n].first;
        const auto o_max = halo_regions[n].second;
        // transform to neighbor coordinates
        const auto t_min = t(o_min[0], o_min[1], c);
        const auto t_max = t(o_max[0], o_max[1], c);
        // get expected coordinates
        const auto e_min = expected_halo_regions[n].first;
        const auto e_max = expected_halo_regions[n].second;
        // compare results
        EXPECT_EQ(t_min, e_min);
        EXPECT_EQ(t_max, e_max);
        // look up inverse transform
        const auto d = reverse_idx_lu(tile, tile_lu[tile][n]);
        const auto& tr = transform_lu[tile_lu[tile][n]][d];
        // inverse-transform transformed coordinates
        const auto r_min = tr(t_min[0], t_min[1], c);
        const auto r_max = tr(t_max[0], t_max[1], c);
        // compare to original coordinates
        EXPECT_EQ(o_min, r_min);
        EXPECT_EQ(o_max, r_max);
        // look up inverse transform directly
        const auto& tr2 = inverse_transform_lu[tile][n];
        // inverse-transform transformed coordinates
        const auto r2_min = tr2(t_min[0], t_min[1], c);
        const auto r2_max = tr2(t_max[0], t_max[1], c);
        // compare to original coordinates
        EXPECT_EQ(o_min, r2_min);
        EXPECT_EQ(o_max, r2_max);
    }
}

TEST(cubed_sphere, transform)
{
    // check tile neighbors for all edges
    neighbor_test(0, 4, 1, 5, 2);
    neighbor_test(1, 0, 3, 5, 2);
    neighbor_test(2, 0, 3, 1, 4);
    neighbor_test(3, 2, 5, 1, 4);
    neighbor_test(4, 2, 5, 3, 0);
    neighbor_test(5, 4, 1, 3, 0);
    // cube size c (number of cells along an edge of the cube)
    for (int c=10; c<30; ++c) {
        // halo size b
        for (int b=1; b<4; ++b) {
            transform_test(c, b, 0);
            transform_test(c, b, 1);
            transform_test(c, b, 2);
            transform_test(c, b, 3);
            transform_test(c, b, 4);
            transform_test(c, b, 5);
        }
    }
}

