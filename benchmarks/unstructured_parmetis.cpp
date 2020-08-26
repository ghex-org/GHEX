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


/** @brief Unstructured exchange benchmark (in-place receive against buffered receive)*/
TEST(unstructured_parmetis, receive_type) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

}
