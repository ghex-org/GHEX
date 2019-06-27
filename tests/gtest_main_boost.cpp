/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#ifndef INCLUDED_GTEST_MAIN_BOOST_CPP
#define INCLUDED_GTEST_MAIN_BOOST_CPP
#include <cstdio>
#include "gtest/gtest.h"
#include <boost/mpi/environment.hpp>


GTEST_API_ int main(int argc, char **argv) {

    int provided;
    int init_res = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    // boost::mpi::environment env(argc, argv);

    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    auto res = RUN_ALL_TESTS();

    MPI_Finalize();

    return res;

}

#endif /* INCLUDED_GTEST_MAIN_BOOST_CPP */
