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
#ifndef INCLUDED_GHEX_GTEST_MAIN_CPP
#define INCLUDED_GHEX_GTEST_MAIN_CPP

#include <fstream>
#include <mpi.h>
#include "gtest/gtest.h"
#include "mpi_listener.hpp"

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <gridtools/common/cuda_util.hpp>
#include <ghex/common/cuda_runtime.hpp>
#endif

GTEST_API_ int main(int argc, char **argv) {

    int required = MPI_THREAD_MULTIPLE;
    int provided;
    int init_result = MPI_Init_thread(&argc, &argv, required, &provided);
    if (init_result == MPI_ERR_OTHER)
        throw std::runtime_error("MPI init failed");
    if (provided < required)
        throw std::runtime_error("MPI does not support required threading level");

    // printf("Running main() from %s\n", __FILE__);
    testing::InitGoogleTest(&argc, argv);

    // set up a custom listener that prints messages in an MPI-friendly way
    auto &listeners = testing::UnitTest::GetInstance()->listeners();
    // first delete the original printer
    delete listeners.Release(listeners.default_result_printer());
    // now add our custom printer
    listeners.Append(new mpi_listener("results_tests"));
    
    //auto tmp = new mpi_listener("results_tests");
    //std::cout << tmp << std::endl;

    // record the local return value for tests run on this mpi rank
    //      0 : success
    //      1 : failure
    auto result = RUN_ALL_TESTS();

    // perform global collective, to ensure that all ranks return
    // the same exit code
    decltype(result) global_result{};
    MPI_Allreduce(&result, &global_result, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);

    MPI_Finalize();

    return global_result;

}

#endif /* INCLUDED_GHEX_GTEST_MAIN_CPP */

