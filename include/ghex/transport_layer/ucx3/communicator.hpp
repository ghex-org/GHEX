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
#ifndef INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP

#include "../communicator.hpp"
#include "./worker.hpp"
#include "./request.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            /** Mpi communicator which exposes basic non-blocking transport functionality and 
              * returns futures to await said transports to complete. */
            template<>
            class communicator<ucx_tag>
            {
            public: // member types

                using rank_type = typename ucx::worker_t::rank_type;
                using tag_type  = typename ucx::worker_t::tag_type;
                using request   = ucx::request;
                using future    = request;

            private: // members

                ucx::worker_t* m_send_worker;
                ucx::worker_t* m_send_worker_ts;
                ucx::worker_t* m_recv_worker_ts;

            public: // ctors

                communicator(ucx::worker_t* send_worker, ucx::worker_t* send_worker_ts, ucx::worker_t* recv_worker_ts)
                : m_send_worker(send_worker)
                , m_send_worker_ts(send_worker_ts)
                , m_recv_worker_ts(recv_worker_ts)
                {}

            public: // member functions

                rank_type rank() const noexcept { return m_send_worker->rank(); }
                rank_type size() const noexcept { return m_send_worker->size(); }

                static void empty_send_callback(void *, ucs_status_t) {}
                static void empty_recv_callback(void *, ucs_status_t, ucp_tag_recv_info_t*) {}
                    
                template<typename Message>
                request send(Message& msg, rank_type dst, tag_type tag)
                {
                    return send(msg.data(), msg.size()*sizeof(typename Message::value_type), dst, tag, m_send_worker, m_recv_worker_ts);
                }

                template<typename Message>
                request send_ts(Message& msg, rank_type dst, tag_type tag)
                {
                    return send(msg.data(), msg.size()*sizeof(typename Message::value_type), dst, tag, m_send_worker_ts, m_recv_worker_ts);
                }

                template<typename Message>
                request recv(Message& msg, rank_type src, tag_type tag)
                {
                    return recv(msg.data(), msg.size()*sizeof(typename Message::value_type), src, tag, m_recv_worker_ts, m_send_worker);
                }

                template<typename Message>
                request recv_ts(Message& msg, rank_type src, tag_type tag)
                {
                    return recv(msg.data(), msg.size()*sizeof(typename Message::value_type), src, tag, m_recv_worker_ts, m_send_worker_ts);
                }

                request send(void* buffer, std::size_t size, rank_type dst, tag_type tag, ucx::worker_t* send_worker, ucx::worker_t* recv_worker)
                {
                    const auto& ep = send_worker->connect(dst);
                    const auto stag = ((std::uint_fast64_t)tag << 32) | 
                                       (std::uint_fast64_t)(rank());
                    ucs_status_ptr_t ret;
                    if (send_worker->m_shared)
                    {
                    //const typename ucx::worker_t::lock_type lock(*(send_worker->m_mutex));
                    send_worker->lock();
                    ret = ucp_tag_send_nb(
                        ep.get(),                                        // destination
                        buffer,                                          // buffer
                        size,                                            // buffer size
                        ucp_dt_make_contig(1),                           // data type
                        stag,                                            // tag
                        &communicator::empty_send_callback);             // callback function pointer: empty here
                    send_worker->unlock();
                    }
                    else
                    {
                    ret = ucp_tag_send_nb(
                        ep.get(),                                        // destination
                        buffer,                                          // buffer
                        size,                                            // buffer size
                        ucp_dt_make_contig(1),                           // data type
                        stag,                                            // tag
                        &communicator::empty_send_callback);             // callback function pointer: empty here
                    }
                    if (reinterpret_cast<std::uintptr_t>(ret) == UCS_OK)
                    {
                        // send operation is completed immediately and the call-back function is not invoked
                        return {nullptr, send_worker, recv_worker};
                    } 
                    else if(!UCS_PTR_IS_ERR(ret))
                    {
                        return {(void*)ret, send_worker, recv_worker};
                    }
                    else
                    {
                        // an error occurred
                        throw std::runtime_error("ghex: ucx error - send operation failed");
                    }
                }

                request recv(void* buffer, std::size_t size, rank_type src, tag_type tag, ucx::worker_t* recv_worker, ucx::worker_t* send_worker)
                {
                    const auto rtag = ((std::uint_fast64_t)tag << 32) | 
                                       (std::uint_fast64_t)(src);
                    ucs_status_ptr_t ret;
                    {
                    //const typename ucx::worker_t::lock_type lock(*(recv_worker->m_mutex));
                    recv_worker->lock();
                    ret = ucp_tag_recv_nb(
                        recv_worker->get(),                              // worker
                        buffer,                                          // buffer
                        size,                                            // buffer size
                        ucp_dt_make_contig(1),                           // data type
                        rtag,                                            // tag
                        ~std::uint_fast64_t(0ul),                        // tag mask
                        &communicator::empty_recv_callback);             // callback function pointer: empty here
                    recv_worker->unlock();
                    }
                    if(!UCS_PTR_IS_ERR(ret))
                    {
                        return {(void*)ret, recv_worker, send_worker};
                    }
                    else
                    {
                        // an error occurred
                        throw std::runtime_error("ghex: ucx error - recv operation failed");
                    }
                }
            };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP

