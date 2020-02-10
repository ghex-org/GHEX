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

                /** @brief completion handle returned from callback based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_cb
                {
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using state_type        = communicator_state<ThreadPrimitives>;
                    using queue_type        = typename state_type::queue_type;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using tag_type          = typename state_type::tag_type;
                    using completion_type   = ::gridtools::ghex::tl::cb::request;

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
                        if(!m_queue) return false;
                        auto res = m_queue->cancel(m_completed.queue_index());
                        if (res)
                        {
                            m_queue = nullptr;
                            m_completed.reset();
                        }
                        return res;
                    }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_REQUEST_CB_HPP */

