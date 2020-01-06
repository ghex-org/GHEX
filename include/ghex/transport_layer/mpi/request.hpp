/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_TL_MPI_REQUEST_HPP
#define INCLUDED_GHEX_TL_MPI_REQUEST_HPP

#include "./error.hpp"
#include "../../common/c_managed_struct.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

                /** @brief thin wrapper around MPI_Request */
                struct request_t
                {
                    GHEX_C_STRUCT(req_type, MPI_Request)
                    req_type m_req = MPI_REQUEST_NULL;

                    void wait()
                    {
                        //MPI_Status status;
                        GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_req.get(), MPI_STATUS_IGNORE));
                    }

                    bool test()
                    {
                        //MPI_Status result;
                        int flag = 0;
                        GHEX_CHECK_MPI_RESULT(MPI_Test(&m_req.get(), &flag, MPI_STATUS_IGNORE));
                        return flag != 0;
                    }

                    operator       MPI_Request&()       noexcept { return m_req; }
                    operator const MPI_Request&() const noexcept { return m_req; }
                          MPI_Request& get()       noexcept { return m_req; }
                    const MPI_Request& get() const noexcept { return m_req; }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_REQUEST_HPP */

