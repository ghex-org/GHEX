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
#ifndef INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP
#define INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP

#include "./error.hpp"
#include "./future.hpp"
#include "../callback_utils.hpp"

namespace gridtools {

    namespace ghex {

        namespace tl {

            namespace mpi {

                template<typename ThreadPrimitives>
                struct shared_communicator_state {
                    using parallel_context_type = parallel_context<ThreadPrimitives>;
                    using transport_context_type = transport_context<mpi_tag, ThreadPrimitives>;
                    using rank_type = int;
                    using tag_type = int;

                    MPI_Comm m_comm;
                    transport_context_type* m_context;
                    parallel_context_type* m_parallel_context;
                    rank_type m_rank;
                    rank_type m_size;

                    shared_communicator_state(MPI_Comm comm, transport_context_type* tc, parallel_context_type* pc)
                    : m_comm{comm}
                    , m_context{tc}
                    , m_parallel_context{pc}
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(comm) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(comm) }
                    {}

                    rank_type rank() const noexcept { return m_rank; }
                    rank_type size() const noexcept { return m_size; }
                };

                template<typename ThreadPrimitives>
                struct communicator_state {
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using parallel_context_type = typename shared_state_type::parallel_context_type;
                    using thread_token = typename parallel_context_type::thread_token;
                    using rank_type = typename shared_state_type::rank_type;
                    using tag_type = typename shared_state_type::tag_type;
                    template<typename T>
                    using future = future_t<T>;
                    using queue_type = ::gridtools::ghex::tl::cb::queue<future<void>, rank_type, tag_type>;

                    thread_token* m_token_ptr;
                    queue_type m_send_queue;
                    queue_type m_recv_queue;

                    communicator_state(thread_token* t)
                    : m_token_ptr{t}
                    {}

                    int progress() { return m_send_queue.progress() + m_recv_queue.progress(); }
                };

            } // namespace mpi

        } // namespace tl

    } // namespace ghex

} //namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP */
