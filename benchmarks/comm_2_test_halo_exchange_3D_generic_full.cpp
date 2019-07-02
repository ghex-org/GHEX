/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */

#include "gtest/gtest.h"
//#define GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI
#include "gtest_main_boost.cpp"

namespace halo_exchange_3D_generic_full {

    
    bool test(int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3) 
    {
        return true;
    }

} // namespace halo_exchange_3D_generic_full

TEST(Communication, comm_2_test_halo_exchange_3D_generic_full) {
    bool passed = halo_exchange_3D_generic_full::test(98, 54, 87, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
    EXPECT_TRUE(passed);
}
// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

