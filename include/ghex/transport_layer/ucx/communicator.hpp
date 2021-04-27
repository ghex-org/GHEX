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
#ifndef INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP

#include <atomic>
#include <mutex>
#include "../shared_message_buffer.hpp"
#include "./future.hpp"
#include "../util/pthread_spin_mutex.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace ucx {

#define GHEX_TAG_BITS                       32
#define GHEX_RANK_BITS                      32
#define GHEX_ANY_SOURCE_MASK                0x0000000000000000ul
#define GHEX_SPECIFIC_SOURCE_MASK           0x00000000fffffffful
#define GHEX_TAG_MASK                       0xffffffff00000000ul

		
                struct communicator
                {
                    using worker_type            = worker_t;
                    using rank_type              = endpoint_t::rank_type;
                    using tag_type               = typename worker_type::tag_type;
                    using request                = request_ft;
                    template<typename T>
                    using future                 = future_t<T>;
                    // needed for now for high-level API
                    using address_type           = rank_type;

                    using request_cb_type        = request_cb;
                    using request_cb_data_type   = typename request_cb_type::data_type;
                    using request_cb_state_type  = typename request_cb_type::state_type;
                    using message_type           = typename request_cb_type::message_type;
                    using progress_status        = gridtools::ghex::tl::cb::progress_status;

                    worker_type*  m_recv_worker;
                    worker_type*  m_send_worker;
                    ucp_worker_h  m_ucp_rw;
                    ucp_worker_h  m_ucp_sw;
                    rank_type     m_rank;
                    rank_type     m_size;

                    communicator(worker_type* rw, worker_type* sw) noexcept
                    : m_recv_worker{rw}
                    , m_send_worker{sw}
                    , m_ucp_rw{rw->get()}
                    , m_ucp_sw{sw->get()}
                    , m_rank{m_send_worker->rank()}
                    , m_size{m_send_worker->size()}
                    {}

                    communicator(const communicator&) = default;
                    communicator(communicator&&) = default;
                    communicator& operator=(const communicator&) = default;
                    communicator& operator=(communicator&&) = default;

                    rank_type rank() const noexcept { return m_rank; }
                    rank_type size() const noexcept { return m_size; }
                    address_type address() const { return rank(); }

                    bool is_local(rank_type r) const noexcept { return m_recv_worker->rank_topology().is_local(r); }
                    rank_type local_rank() const noexcept { return m_recv_worker->rank_topology().local_rank(); }
                    auto mpi_comm() const noexcept { return m_recv_worker->rank_topology().mpi_comm(); }

                    /** @brief send a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a meassage type
                     * @param msg an l-value reference to the message to be sent
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template <typename Message>
                    [[nodiscard]] future<void> send(const Message &msg, rank_type dst, tag_type tag)
                    {
                        const auto& ep = m_send_worker->connect(dst);
                        const auto stag = ((std::uint_fast64_t)tag << GHEX_TAG_BITS) |
                                           (std::uint_fast64_t)(rank());
                        auto ret = ucp_tag_send_nb(
                            ep.get(),                                        // destination
                            msg.data(),                                      // buffer
                            msg.size()*sizeof(typename Message::value_type), // buffer size
                            ucp_dt_make_contig(1),                           // data type
                            stag,                                            // tag
                            &communicator::empty_send_callback);             // callback function pointer: empty here

                        if (reinterpret_cast<std::uintptr_t>(ret) == UCS_OK)
                        {
                            // send operation is completed immediately and the call-back function is not invoked
                            return request{nullptr};
                        }
                        else if(!UCS_PTR_IS_ERR(ret))
                        {
                            return request{request::data_type::construct(ret, m_recv_worker, m_send_worker, request_kind::send)};
                        }
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - send operation failed");
                        }
                    }

                    /** @brief receive a message. The message must be kept alive by the caller until the communication is
                     * finished.
                     * @tparam Message a meassage type
                     * @param msg an l-value reference to the message to be sent
                     * @param src the source rank
                     * @param tag the communication tag
                     * @return a future to test/wait for completion */
                    template <typename Message>
                    [[nodiscard]] future<void> recv(Message &msg, rank_type src, tag_type tag)
                    {
                        const auto rtag = (GHEX_ANY_SOURCE == src) ?
			    ((std::uint_fast64_t)tag << GHEX_TAG_BITS) :
			    ((std::uint_fast64_t)tag << GHEX_TAG_BITS) | (std::uint_fast64_t)(src);
			const auto rtag_mask = (GHEX_ANY_SOURCE == src) ?
			    (GHEX_TAG_MASK | GHEX_ANY_SOURCE_MASK) :
			    (GHEX_TAG_MASK | GHEX_SPECIFIC_SOURCE_MASK);
                        std::lock_guard<worker_type::mutex_t> lock(m_send_worker->mutex());
                                auto ret = ucp_tag_recv_nb(
                                    m_recv_worker->get(),                            // worker
                                    msg.data(),                                      // buffer
                                    msg.size()*sizeof(typename Message::value_type), // buffer size
                                    ucp_dt_make_contig(1),                           // data type
                                    rtag,                                            // tag
                                    rtag_mask,                                       // tag mask
                                    &communicator::empty_recv_callback);             // callback function pointer: empty here
                                if(!UCS_PTR_IS_ERR(ret))
                                {
                                    if (UCS_INPROGRESS != ucp_request_check_status(ret))
                                    {
                                        // recv completed immediately
                                        // we need to free the request here, not in the callback
                                        auto ucx_ptr = ret;
                                        request_init(ucx_ptr);
                                        ucp_request_free(ucx_ptr);
                                        return request{nullptr};
                                    }
                                    else
                                    {
                                        return request{request::data_type::construct(ret, m_recv_worker, m_send_worker, request_kind::recv)};
                                    }
                                }
                                else
                                {
                                    // an error occurred
                                    throw std::runtime_error("ghex: ucx error - recv operation failed");
                                }
                    }

                    /** @brief Function to poll the transport layer and check for completion of operations with an
                      * associated callback. When an operation completes, the corresponfing call-back is invoked
                      * with the message, rank and tag associated with this communication.
                      * @return non-zero if any communication was progressed, zero otherwise. */
                    progress_status progress()
                    {
                        gridtools::ghex::tl::cb::progress_status status;
                        int p = 0, c;
			
			while((c = ucp_worker_progress(m_ucp_sw))) p+=c;

                        /* this is really important for large-scale multithreading */
                        sched_yield();

                        status.m_num_sends = std::exchange(m_send_worker->m_progressed_sends, 0);
                        {
                        std::lock_guard<worker_type::mutex_t> lock(m_send_worker->mutex());
                        int c;
                        while((c = ucp_worker_progress(m_ucp_rw))) p+=c;
                        status.m_num_recvs = std::exchange(m_recv_worker->m_progressed_recvs, 0);
                        status.m_num_cancels = std::exchange(m_recv_worker->m_progressed_cancels, 0);
                        }
                        return status;
                    }

                   /** @brief send a message and get notified with a callback when the communication has finished.
                     * The ownership of the message is transferred to this communicator and it is safe to destroy the
                     * message at the caller's site.
                     * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                     * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                     * @param msg r-value reference to any_message instance
                     * @param dst the destination rank
                     * @param tag the communication tag
                     * @param callback a callback instance
                     * @return a request to test (but not wait) for completion */
                    template<typename CallBack>
                    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        const auto& ep = m_send_worker->connect(dst);
                        const auto stag = ((std::uint_fast64_t)tag << GHEX_TAG_BITS) |
                                           (std::uint_fast64_t)(rank());
                        auto ret = ucp_tag_send_nb(
                            ep.get(),                                        // destination
                            msg.data(),                                      // buffer
                            msg.size(),                                      // buffer size
                            ucp_dt_make_contig(1),                           // data type
                            stag,                                            // tag
                            &communicator::send_callback);                   // callback function pointer

                        if (reinterpret_cast<std::uintptr_t>(ret) == UCS_OK)
                        {
                            // send operation is completed immediately and the call-back function is not invoked
                            // call the callback
                            callback(std::move(msg), dst, tag);
                            ++(m_send_worker->m_progressed_sends);
                            return {};
                        }
                        else if(!UCS_PTR_IS_ERR(ret))
                        {
                            auto req_ptr = request_cb_data_type::construct(ret,
                                m_send_worker,
                                request_kind::send,
                                std::move(msg),
                                dst,
                                tag,
                                std::forward<CallBack>(callback),
                                std::make_shared<request_cb_state_type>(false));
                            return {req_ptr, req_ptr->m_completed};
                        }
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - send operation failed");
                        }
                    }

                   /** @brief receive a message and get notified with a callback when the communication has finished.
                     * The ownership of the message is transferred to this communicator and it is safe to destroy the
                     * message at the caller's site.
                     * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                     * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                     * @param msg r-value reference to any_message instance
                     * @param src the source rank
                     * @param tag the communication tag
                     * @param callback a callback instance
                     * @return a request to test (but not wait) for completion */
                    template<typename CallBack>
                    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        const auto rtag = (GHEX_ANY_SOURCE == src) ?
			    ((std::uint_fast64_t)tag << GHEX_TAG_BITS) :
			    ((std::uint_fast64_t)tag << GHEX_TAG_BITS) | (std::uint_fast64_t)(src);
			const auto rtag_mask = (GHEX_ANY_SOURCE == src) ?
			    (GHEX_TAG_MASK | GHEX_ANY_SOURCE_MASK) :
			    (GHEX_TAG_MASK | GHEX_SPECIFIC_SOURCE_MASK);
                        std::lock_guard<worker_type::mutex_t> lock(m_send_worker->mutex());
                        auto ret = ucp_tag_recv_nb(
                                    m_ucp_rw,                                        // worker
                                    msg.data(),                                      // buffer
                                    msg.size(),                                      // buffer size
                                    ucp_dt_make_contig(1),                           // data type
                                    rtag,                                            // tag
                                    rtag_mask,                                       // tag mask
                                    &communicator::recv_callback);                   // callback function pointer
                                if(!UCS_PTR_IS_ERR(ret))
                                {
                                    if (UCS_INPROGRESS != ucp_request_check_status(ret))
                                    {
                                        // early completed
                                        callback(std::move(msg), src, tag);
                                        ++(m_recv_worker->m_progressed_recvs);
                                        // we need to free the request here, not in the callback
                                        auto ucx_ptr = ret;
                                        request_cb_data_type::get(ucx_ptr).m_kind = request_kind::none;
                                        ucp_request_free(ucx_ptr);
                                        return request_cb_type{};
                                    }
                                    else
                                    {
                                        auto req_ptr = request_cb_data_type::construct(ret,
                                            m_recv_worker,
                                            request_kind::recv,
                                            std::move(msg),
                                            src,
                                            tag,
                                            std::forward<CallBack>(callback),
                                            std::make_shared<request_cb_state_type>(false));
                                        return request_cb_type{req_ptr, req_ptr->m_completed};
                                    }
                                }
                                else
                                {
                                    // an error occurred
                                    throw std::runtime_error("ghex: ucx error - recv operation failed");
                                }
                    }

                private:

                    static void empty_send_callback(void *, ucs_status_t) {}

                    static void empty_recv_callback(void *, ucs_status_t, ucp_tag_recv_info_t*) {}

                    inline static void send_callback(void * __restrict ucx_req, ucs_status_t status)
                    {
                        auto& req = request_cb_data_type::get(ucx_req);
                        if (status == UCS_OK) {
                            // call the callback
                            req.m_cb(std::move(req.m_msg), req.m_rank, req.m_tag);
                            ++(req.m_worker->m_progressed_sends);
                        }
                        // else: cancelled - do nothing - cancel for sends does not exist
                        // set completion bit
                        *req.m_completed = true;
                        // destroy the request - releases the message
                        req.m_kind = request_kind::none;
                        req.~request_cb_data_type();
                        // free ucx request
                        ucp_request_free(ucx_req);
                    }

                    // this function must be called from within a locked region
                    inline static void recv_callback(void * __restrict ucx_req, ucs_status_t status, ucp_tag_recv_info_t*info /*info*/)
                    {
                        auto& req = request_cb_data_type::get(ucx_req);

                        if (status == UCS_OK)
                        {
                            if (static_cast<int>(req.m_kind) == 0)
                            {
                                // we're in early completion mode
                                return;
                            }
                            
                            req.m_cb(std::move(req.m_msg), info->sender_tag & GHEX_SPECIFIC_SOURCE_MASK, req.m_tag);
                            ++(req.m_worker->m_progressed_recvs);
                            // set completion bit
                            *req.m_completed = true;
                            // destroy the request - releases the message
                            req.m_kind = request_kind::none;
                            req.~request_cb_data_type();
                            // free ucx request
                            ucp_request_free(ucx_req);
                        }
                        else if (status == UCS_ERR_CANCELED)
                        {
                            // canceled - do nothing
                            ++(req.m_worker->m_progressed_cancels);
                            // set completion bit
                            *req.m_completed = true;
                            // destroy the request - releases the message
                            req.m_kind = request_kind::none;
                            req.~request_cb_data_type();
                            // free ucx request
                            ucp_request_free(ucx_req);
                        }
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - recv message truncated");
                        }
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP */
