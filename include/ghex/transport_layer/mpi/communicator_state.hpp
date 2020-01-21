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
                struct communicator_state {
                    using parallel_context_type = parallel_context<ThreadPrimitives>;
                    using thread_token = typename parallel_context_type::thread_token;
                    using transport_context_type = transport_context<mpi_tag, ThreadPrimitives>;
                    using rank_type = int;
                    using tag_type = int;
                    template<typename T>
                    using future = future_t<T>;
                    using queue_type = ::gridtools::ghex::tl::cb::queue<future<void>, rank_type, tag_type>;

                    MPI_Comm m_comm;
                    transport_context_type* m_context;
                    parallel_context_type* m_parallel_context;
                    thread_token* m_token_ptr;
                    queue_type m_queue;

                    communicator_state(MPI_Comm comm, transport_context_type* tc, parallel_context_type* pc, thread_token* t)
                    : m_comm{comm}
                    , m_context{tc}
                    , m_parallel_context{pc}
                    , m_token_ptr{t}
                    {}

                    rank_type rank() const noexcept {
                        rank_type r;
                        MPI_Comm_rank(m_comm, &r);
                        return r;
                    }

                    rank_type size() const noexcept {
                        rank_type s;
                        MPI_Comm_size(m_comm, &s);
                        return s;
                    }

                    int progress() { return m_queue.progress(); }
                };

            } // namespace mpi

        } // namespace tl

    } // namespace ghex

} //namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP */
