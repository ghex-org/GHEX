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
#ifndef INCLUDED_GHEX_TL_MPI_REQUEST_CB_HPP
#define INCLUDED_GHEX_TL_MPI_REQUEST_CB_HPP

#include "./request.hpp"
#include "../context.hpp"
#include "./communicator_state.hpp"
#include "../callback_utils.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

                template<typename ThreadPrimitives>
                struct request_cb
                {
                    using comm_state_type   = communicator_state<ThreadPrimitives>;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using tag_type          = typename comm_state_type::tag_type;
                    //using state_type        = //bool;
                    using completion_type   = ::gridtools::ghex::tl::cb::request;

                    comm_state_type* m_comm_state = nullptr;
                    completion_type m_completed;
                    
                    bool test()
                    {
                        if(!m_comm_state) return true;
                        if (m_completed.is_ready())
                        {
                            m_comm_state = nullptr;
                            m_completed.reset();
                            return true;
                        }
                        return false;
                    }

                    bool cancel()
                    {
                        return true;
                    }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_REQUEST_CB_HPP */

