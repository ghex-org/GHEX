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

#include <ghex/mpi/error.hpp>

namespace ghex
{
namespace mpi
{
/** @brief thin wrapper around MPI_Request */
struct request
{
    MPI_Request m_req = MPI_REQUEST_NULL;

    void wait() { GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_req, MPI_STATUS_IGNORE)); }

    bool test()
    {
        int flag = 0;
        GHEX_CHECK_MPI_RESULT(MPI_Test(&m_req, &flag, MPI_STATUS_IGNORE));
        return flag != 0;
    }

    operator MPI_Request() const noexcept { return m_req; }

    MPI_Request&       get() noexcept { return m_req; }
    const MPI_Request& get() const noexcept { return m_req; }
};

} // namespace mpi
} // namespace ghex
