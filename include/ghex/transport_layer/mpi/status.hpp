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
#ifndef INCLUDED_GHEX_TL_MPI_STATUS_HPP
#define INCLUDED_GHEX_TL_MPI_STATUS_HPP

#include "./error.hpp"
#include "../../common/c_managed_struct.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

                /** @brief thin wrapper around MPI_Request */
                struct status_t
                {
                    GHEX_C_STRUCT(stat_type, MPI_Status)
                    stat_type m_status;

                    int source() const noexcept { return m_status.get().MPI_SOURCE; }

                    int tag() const noexcept { return m_status.get().MPI_TAG; }

                    int error() const noexcept { return m_status.get().MPI_ERROR; }

                    bool cancelled() const
                    {
                        int flag = 0;
                        GHEX_CHECK_MPI_RESULT(MPI_Test_cancelled(&m_status.get(), &flag));
                        return flag != 0;
                    }

                    operator       MPI_Status&()       noexcept { return m_status; }
                    operator const MPI_Status&() const noexcept { return m_status; }
                          MPI_Status& get()       noexcept { return m_status; }
                    const MPI_Status& get() const noexcept { return m_status; }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_STATUS_HPP */

