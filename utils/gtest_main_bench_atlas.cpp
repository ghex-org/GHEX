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
#ifndef INCLUDED_GHEX_GTEST_MAIN_ATLAS_CPP
#define INCLUDED_GHEX_GTEST_MAIN_ATLAS_CPP

#include <fstream>
#include <mpi.h>

#include <gtest/gtest.h>

#include <atlas/library/Library.h>

#include <gridtools/tools/mpi_unit_test_driver/mpi_listener.hpp>


GTEST_API_ int main(int argc, char** argv) {

    MPI_Init(&argc, &argv);

    // printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);

    atlas::Library::instance().initialise(argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_benchmarks"));

    // record the local return value for tests run on this mpi rank
    //      0 : success
    //      1 : failure
    auto result = RUN_ALL_TESTS();

    // perform global collective, to ensure that all ranks return the same exit code
    decltype(result) global_result{};
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    atlas::Library::instance().finalise();

    MPI_Finalize();

    return global_result;

}

#endif /* INCLUDED_GHEX_GTEST_MAIN_ATLAS_CPP */
