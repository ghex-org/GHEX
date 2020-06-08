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
#ifndef INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP
#define INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP

#include "./error.hpp"
#include "./future.hpp"
#include "../callback_utils.hpp"

namespace gridtools {

    namespace ghex {

        namespace tl {

            namespace mpi {

                /** @brief common data which is shared by all communicators. This class is thread safe.
                 */
                struct shared_communicator_state {
                    using transport_context_type = transport_context<mpi_tag>;
                    using rank_type = int;
                    using tag_type = int;

                    MPI_Comm m_comm;
                    transport_context_type* m_context;
                    rank_type m_rank;
                    rank_type m_size;

                    shared_communicator_state(MPI_Comm comm, transport_context_type* tc)
                    : m_comm{comm}
                    , m_context{tc}
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(comm) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(comm) }
                    {}

                    rank_type rank() const noexcept { return m_rank; }
                    rank_type size() const noexcept { return m_size; }
                };

                /** @brief communicator per-thread data.
                 */
                struct communicator_state {
                    using shared_state_type = shared_communicator_state;
                    using rank_type = typename shared_state_type::rank_type;
                    using tag_type = typename shared_state_type::tag_type;
                    template<typename T>
                    using future = future_t<T>;
                    using queue_type = ::gridtools::ghex::tl::cb::callback_queue<future<void>, rank_type, tag_type>;
                    using progress_status = gridtools::ghex::tl::cb::progress_status;

                    queue_type m_send_queue;
                    queue_type m_recv_queue;
                    int  m_progressed_sends = 0;
                    int  m_progressed_recvs = 0;

                    communicator_state() = default;

                    progress_status progress() {
                        m_progressed_sends += m_send_queue.progress();
                        m_progressed_recvs += m_recv_queue.progress();
                        return {
                            std::exchange(m_progressed_sends,0),
                            std::exchange(m_progressed_recvs,0),
                            std::exchange(m_recv_queue.m_progressed_cancels,0)};
                    }
                };

            } // namespace mpi

        } // namespace tl

    } // namespace ghex

} //namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_STATE_HPP */
