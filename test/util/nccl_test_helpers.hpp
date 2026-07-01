/*
 * ghex-org
 *
 * Copyright (c) 2014-2026, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gtest/gtest.h>
#include <mpi.h>
#include <stdexcept>
#include <string>

#include <ghex/context.hpp>

namespace ghex::test
{
inline bool
is_nccl_backend()
{
    static const bool result =
        ghex::context(MPI_COMM_WORLD, false).transport_context()->get_transport_option("name") ==
        std::string("nccl");
    return result;
}

inline void
handle_nccl_thread_safe_exception(std::runtime_error const& e)
{
    if (is_nccl_backend()) { EXPECT_STREQ(e.what(), "NCCL not supported with thread_safe = true"); }
    else { throw; }
}

inline void
handle_nccl_self_comm_exception(std::runtime_error const& e)
{
    if (is_nccl_backend())
    {
        EXPECT_STREQ(e.what(), "oomph NCCL backend: self-send/recv requires an active NCCL group. "
                               "Use start_group()/end_group() around self-send/recv operations.");
    }
    else { throw; }
}
} // namespace ghex::test
