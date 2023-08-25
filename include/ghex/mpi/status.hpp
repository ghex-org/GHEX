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
struct status
{
    MPI_Status m_status;

    int source() const noexcept { return m_status.MPI_SOURCE; }

    int tag() const noexcept { return m_status.MPI_TAG; }

    int error() const noexcept { return m_status.MPI_ERROR; }

    bool cancelled() const
    {
        int flag = 0;
        GHEX_CHECK_MPI_RESULT(MPI_Test_cancelled(&m_status, &flag));
        return flag != 0;
    }

    operator MPI_Status() const noexcept { return m_status; }

    MPI_Status&       get() noexcept { return m_status; }
    const MPI_Status& get() const noexcept { return m_status; }
};

} // namespace mpi
} // namespace ghex
