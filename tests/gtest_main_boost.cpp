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

    boost::mpi::environment env(argc, argv);

    printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();

}

#endif /* INCLUDED_GTEST_MAIN_BOOST_CPP */
