/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <string>
#include <stdexcept>
#include <mpi.h>

#ifdef NDEBUG
#define GHEX_CHECK_MPI_RESULT(x) x;
#else
#define GHEX_CHECK_MPI_RESULT(x)                                                                   \
    if (x != MPI_SUCCESS)                                                                          \
        throw std::runtime_error("GHEX Error: MPI Call failed " + std::string(#x) + " in " +       \
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif
