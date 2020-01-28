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
#ifndef INCLUDED_GHEX_TL_UCX_REQUEST_HPP
#define INCLUDED_GHEX_TL_UCX_REQUEST_HPP

#include <functional>
#include "../context.hpp"
#include "./worker.hpp"
#include "../callback_utils.hpp"
#include "../../threads/atomic/primitives.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

                /** @brief the type of the communication */
                enum class request_kind : int { none=0, send, recv };

                /** @brief data required for future based communication which will be stored in ucx provided storage.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_data_ft
                {
                    using worker_type            = worker_t<ThreadPrimitives>;

                    void*        m_ucx_ptr;
                    worker_type* m_recv_worker;
                    worker_type* m_send_worker;
                    request_kind m_kind;

                    /** @brief alignment mask */
                    static constexpr std::uintptr_t mask = ~(alignof(request_data_ft)-1u);

                    /** @brief construct the struct inplace.
                      * @tparam Args constructor argument types
                      * @param ptr ucx provided pointer to storage
                      * @param ...args constructor arguments
                      * @return pointer to the instance (may be different from ucx pointer due to alignment) */
                    template<typename... Args>
                    static request_data_ft* construct(void* ptr, Args&& ...args)
                    {
                        // align pointer
                        auto a_ptr = reinterpret_cast<request_data_ft*>
                        ((reinterpret_cast<std::uintptr_t>((unsigned char*)ptr) + alignof(request_data_ft)-1) & mask);
                        new(a_ptr) request_data_ft{ptr,std::forward<Args>(args)...};
                        return a_ptr;
                    }
                };
                using request_data_size_ft = std::integral_constant<std::size_t,
                sizeof(request_data_ft<::gridtools::ghex::threads::atomic::primitives>) +
                alignof(request_data_ft<::gridtools::ghex::threads::atomic::primitives>)>;

                /** @brief data required for callback based communication which will be stored in ucx provided storage.
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_data_cb
                {
                    using worker_type       = worker_t<ThreadPrimitives>;
                    using message_type      = ::gridtools::ghex::tl::cb::any_message;
                    using rank_type         = endpoint_t::rank_type;
                    using tag_type          = typename worker_type::tag_type;
                    using state_type        = bool;

                    void*        m_ucx_ptr;
                    worker_type* m_worker;
                    request_kind m_kind;
                    message_type m_msg;
                    rank_type    m_rank;
                    tag_type     m_tag;
                    std::function<void(message_type, rank_type, tag_type)> m_cb;
                    std::shared_ptr<state_type> m_completed;

                    /** @brief alignment mask */
                    static constexpr std::uintptr_t mask = ~(alignof(request_data_cb)-1u);

                    /** @brief construct the struct inplace.
                      * @tparam Args constructor argument types
                      * @param ptr ucx provided pointer to storage
                      * @param ...args constructor arguments
                      * @return pointer to the instance (may be different from ucx pointer due to alignment) */
                    template<typename... Args>
                    static request_data_cb* construct(void* __restrict ptr, Args&& ...args)
                    {
                        // align pointer
                        auto a_ptr = reinterpret_cast<request_data_cb*>
                        ((reinterpret_cast<std::uintptr_t>((unsigned char*)ptr) + alignof(request_data_cb)-1) & mask);
                        new(a_ptr) request_data_cb{ptr,std::forward<Args>(args)...};
                        return a_ptr;
                    }

                    /** @brief return pointer to an instance from ucx provided storage pointer */
                    static request_data_cb& get(void* __restrict ptr)
                    {
                        unsigned char* __restrict cptr = (unsigned char*)ptr;
                        return *reinterpret_cast<request_data_cb*>
                        ((reinterpret_cast<std::uintptr_t>(cptr) + alignof(request_data_cb)-1) & mask);
                    }
                };
                using request_data_size_cb = std::integral_constant<std::size_t,
                sizeof(request_data_cb<::gridtools::ghex::threads::atomic::primitives>) +
                alignof(request_data_cb<::gridtools::ghex::threads::atomic::primitives>)>;

                /** @brief minimum size of the ucx provided storage in bytes */
                using request_data_size = request_data_size_cb;
                
                /** @brief make the ucx provided storage zero */
                inline void request_init(void *req) { std::memset(req, 0, request_data_size::value); }

                /** @brief completion handle returned from future based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_ft
                {
                    using data_type = request_data_ft<ThreadPrimitives>;

                    data_type* m_req = nullptr;

                    request_ft() = default;
                    request_ft(data_type* ptr) noexcept : m_req{ptr} {}
                    request_ft(const request_ft&) = delete;
                    request_ft& operator=(const request_ft&) = delete;

                    request_ft(request_ft&& other) noexcept
                    : m_req{ std::exchange(other.m_req, nullptr) }
                    {}

                    request_ft& operator=(request_ft&& other) noexcept
                    {
                        if (m_req) destroy();
                        m_req = std::exchange(other.m_req, nullptr);
                        return *this;
                    }

                    ~request_ft()
                    {
                        if (m_req) destroy();
                    }

                    void destroy()
                    {
                        void* ucx_ptr = m_req->m_ucx_ptr;
                        m_req->m_send_worker->m_thread_primitives->critical(
                            [ucx_ptr]()
                            {
                                request_init(ucx_ptr);
                                ucp_request_free(ucx_ptr);
                            }
                        );
                    }

                    bool test()
                    {
                        if (!m_req) return true;

                        ucp_worker_progress(m_req->m_send_worker->get());
                        ucp_worker_progress(m_req->m_send_worker->get());
                        ucp_worker_progress(m_req->m_send_worker->get());

                        return m_req->m_send_worker->m_thread_primitives->critical([this]() {
                            ucp_worker_progress(m_req->m_recv_worker->get());

                            // TODO sometimes causes a slowdown, e.g., in the ft_avail
                            // test with 16 threads
                            ucp_worker_progress(m_req->m_recv_worker->get());

                            // check request status
                            // TODO check whether ucp_request_check_status has to be locked also:
                            // it does access the worker!
                            // TODO are we allowed to call this?
                            // m_req might be a send request submitted on another thread, and hence might access
                            // the other threads's send worker...
                            if (UCS_INPROGRESS != ucp_request_check_status(m_req->m_ucx_ptr)) {
                                auto ucx_ptr = m_req->m_ucx_ptr;
                                request_init(ucx_ptr);
                                ucp_request_free(ucx_ptr);
                                m_req = nullptr;
                                return true;
                            }
                            else
                                return false;
                        });
                    }

                    void wait()
                    {
                        if (!m_req) return;
                        while (!test());
                    }

                    bool cancel()
                    {
                        if (!m_req) return true;

                        // TODO at this time, send requests cannot be canceled
                        // in UCX (1.7.0rc1)
                        // https://github.com/openucx/ucx/issues/1162
                        //
                        // TODO the below is only correct for recv requests,
                        // or for send requests under the assumption that
                        // the requests cannot be moved between threads.
                        //
                        // For the send worker we do not use locks, hence
                        // if request is canceled on another thread, it might
                        // clash with another send being submitted by the owner
                        // of ucp_worker

                        if (m_req->m_kind == request_kind::send) return false;

                        m_req->m_send_worker->m_thread_primitives->critical([this]() {
                            auto ucx_ptr = m_req->m_ucx_ptr;
                            auto worker = m_req->m_recv_worker->get();
                            ucp_request_cancel(worker, ucx_ptr);
                        });
                        // wait for the request to either complete, or be canceled
                        wait();
                        return true;
                    }
                };

                /** @brief completion handle returned from callback based communications
                  * @tparam ThreadPrimitives The thread primitives type */
                template<typename ThreadPrimitives>
                struct request_cb
                {
                    using data_type    = request_data_cb<ThreadPrimitives>;
                    using state_type   = typename data_type::state_type;
                    using message_type = typename data_type::message_type;

                    data_type*                  m_req = nullptr;
                    std::shared_ptr<state_type> m_completed;

                    request_cb() = default;
                    request_cb(data_type* ptr, std::shared_ptr<state_type> sp) noexcept : m_req{ptr}, m_completed{sp} {}
                    request_cb(const request_cb&) = delete;
                    request_cb& operator=(const request_cb&) = delete;

                    request_cb(request_cb&& other) noexcept
                    : m_req{ std::exchange(other.m_req, nullptr) }
                    , m_completed{std::move(other.m_completed)}
                    {}

                    request_cb& operator=(request_cb&& other) noexcept
                    {
                        m_req = std::exchange(other.m_req, nullptr);
                        m_completed = std::move(other.m_completed);
                        return *this;
                    }

                    bool test()
                    {
                        if(!m_req) return true;
                        if (*m_completed)
                        {
                            m_req = nullptr;
                            m_completed.reset();
                            return true;
                        }
                        return false;
                    }

                    bool cancel()
                    {
                        // TODO: fix a race. we can only call critical through m_req, but it can be
                        // set to NULL between when we check below, and when we call the critical region.
                        if (!m_req) return true;

                        // TODO at this time, send requests cannot be canceled
                        // in UCX (1.7.0rc1)
                        // https://github.com/openucx/ucx/issues/1162
                        //
                        // TODO the below is only correct for recv requests,
                        // or for send requests under the assumption that
                        // the requests cannot be moved between threads.
                        //
                        // For the send worker we do not use locks, hence
                        // if request is canceled on another thread, it might
                        // clash with another send being submitted by the owner
                        // of ucp_worker

                        if (m_req->m_kind == request_kind::send) return false;

                        return m_req->m_worker->m_thread_primitives->critical([this]() {
                            if (!(*m_completed)) {
                                auto ucx_ptr = m_req->m_ucx_ptr;
                                auto worker = m_req->m_worker->get();

                                // set to zero here????
                                // if we assume that the callback is always called, we
                                // can handle this in the callback body- otherwise needs
                                // to be done here:
                                //request_init(ucx_ptr);
                                ucp_request_cancel(worker, ucx_ptr);
                            }
                            m_req = nullptr;
                            m_completed.reset();
                            return true;
                        });
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */
