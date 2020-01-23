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
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using state_type        = communicator_state<ThreadPrimitives>;
                    using queue_type        = typename state_type::queue_type;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using tag_type          = typename state_type::tag_type;
                    using completion_type   = ::gridtools::ghex::tl::cb::request;

                    //shared_state_type* m_shared_state = nullptr;
                    //state_type* m_state = nullptr;
                    queue_type* m_queue = nullptr;
                    completion_type m_completed;
                    
                    bool test()
                    {
                        if(!m_queue) return true;
                        if (m_completed.is_ready())
                        {
                            m_queue = nullptr;
                            m_completed.reset();
                            return true;
                        }
                        return false;
                    }

                    bool cancel()
                    {
                        return m_queue->cancel(m_completed.m_request_state->m_index);
                    }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_REQUEST_CB_HPP */
