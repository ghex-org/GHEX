/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <ghex/context.hpp>
#include <gtest/gtest.h>
#include "./mpi_runner/mpi_test_fixture.hpp"
#include <iostream>


TEST_F(mpi_test_fixture, context)
{
    using namespace ghex;

    context ctxt(world, thread_safe);
}


