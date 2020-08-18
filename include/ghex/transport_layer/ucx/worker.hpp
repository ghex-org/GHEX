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
#ifndef INCLUDED_GHEX_TL_UCX_WORKER_HPP
#define INCLUDED_GHEX_TL_UCX_WORKER_HPP

#include <map>
#include <deque>
#include <unordered_map>
#include "../../common/moved_bit.hpp"
#include "./error.hpp"
#include "./endpoint.hpp"
#include "../context.hpp"
#include "../util/pthread_spin_mutex.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            // forward declaration
            template<>
            struct transport_context<ucx_tag>;

            namespace ucx {

                struct worker_t
                {
                    using rank_type = typename endpoint_t::rank_type;
                    using tag_type  = int;

                    struct ucp_worker_handle
                    {
                        ucp_worker_h m_worker;
                        moved_bit m_moved;

                        ucp_worker_handle() noexcept : m_moved{true} {}
                        ucp_worker_handle(const ucp_worker_handle&) = delete;
                        ucp_worker_handle& operator=(const ucp_worker_handle&) = delete;
                        ucp_worker_handle(ucp_worker_handle&& other) noexcept = default;

                        ucp_worker_handle& operator=(ucp_worker_handle&& other) noexcept
                        {
                            destroy();
                            m_worker.~ucp_worker_h();
                            ::new((void*)(&m_worker)) ucp_worker_h{other.m_worker};
                            m_moved = std::move(other.m_moved);
                            return *this;
                        }

                        ~ucp_worker_handle() { destroy(); }

                        static void empty_send_cb(void*, ucs_status_t) {}

                        void destroy() noexcept
                        {
                            if (!m_moved)
                                ucp_worker_destroy(m_worker);
                        }

                        operator ucp_worker_h() const noexcept { return m_worker; }
                        ucp_worker_h& get()       noexcept { return m_worker; }
                        const ucp_worker_h& get() const noexcept { return m_worker; }
                    };

                    using cache_type             = std::unordered_map<rank_type, endpoint_t>;
                    using transport_context_type = transport_context<ucx_tag>;
                    using mutex_t = pthread_spin::recursive_mutex;

                    transport_context_type* m_context =  nullptr;
                    rank_type               m_rank;
                    rank_type               m_size;
                    ucp_worker_handle       m_worker;
                    address_t               m_address;
                    cache_type              m_endpoint_cache;
                    int                     m_progressed_sends = 0;
                    mutex_t*                m_mutex_ptr = nullptr;
                    volatile int            m_progressed_recvs = 0;
                    volatile int            m_progressed_cancels = 0;

                    worker_t() = default;
                    worker_t(transport_context_type* c, mutex_t& mm, ucs_thread_mode_t mode);
                    worker_t(const worker_t&) = delete;
                    worker_t(worker_t&& other) noexcept = default;
                    worker_t& operator=(const worker_t&) = delete;
                    worker_t& operator=(worker_t&&) noexcept = default;

                    rank_type rank() const noexcept { return m_rank; }
                    rank_type size() const noexcept { return m_size; }
                    transport_context_type const& context() const noexcept { return *m_context; }
                    inline ucp_worker_h get() const noexcept { return m_worker.get(); }
                    address_t address() const noexcept { return m_address; }
                    inline const endpoint_t& connect(rank_type rank);
                    mutex_t& mutex() { return *m_mutex_ptr; }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_WORKER_HPP */
