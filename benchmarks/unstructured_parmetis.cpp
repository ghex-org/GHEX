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

// TO DO: headers for test code start here

#include <vector>

// TO DO: headers for test code end here

#include <fstream>

#include <gtest/gtest.h>

#include <parmetis.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
// include communication object header file


#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif


template<typename T>
char* as_bytes(T& i) {
    return reinterpret_cast<char*>(&i);
}


/** @brief Unstructured exchange benchmark (in-place receive against buffered receive)*/
TEST(unstructured_parmetis, receive_type) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    std::ifstream ap_fs("Ap.out", std::ios_base::binary);
    std::ifstream ai_fs("Ai.out", std::ios_base::binary);

    // TO DO: sample code starts here (to be removed)

    std::vector<long long unsigned int> ap{};
    std::vector<long long unsigned int> ai{};

    for (long long unsigned int i; ap_fs.read(as_bytes(i), sizeof(i));) {
        ap.push_back(i);
    }

    for (long long unsigned int i; ai_fs.read(as_bytes(i), sizeof(i));) {
        ai.push_back(i);
    }

    // TO DO: sample code ends here (to be removed)


}
