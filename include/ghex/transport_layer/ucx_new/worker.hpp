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

#pragma once

#include <vector>
#include <mutex>
#include <new>
#include <utility>
#include <boost/pool/pool.hpp>
#include <unordered_map>
#include "./request_data.hpp"
#include "./address_db.hpp"

namespace gridtools{
namespace ghex {
namespace tl {
namespace ucx {

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

    void destroy() noexcept
    {
        if (!m_moved)
            ucp_worker_destroy(m_worker);
    }

    ucp_worker_h get() const noexcept { return m_worker; }
};

struct worker
{
    using rank_type = typename address_db_t::rank_type;
    using tag_type = typename address_db_t::tag_type;

    ucp_worker_handle m_worker;
    address_t m_address;

    worker(ucp_context_h context)
    {
        ucp_worker_params_t params;
        params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
        params.thread_mode = UCS_THREAD_MODE_SINGLE;
        GHEX_CHECK_UCX_RESULT(
            ucp_worker_create (context, &params, &m_worker.m_worker)
        );
        ucp_address_t* worker_address;
        std::size_t address_length;
        GHEX_CHECK_UCX_RESULT(
            ucp_worker_get_address(m_worker.get(), &worker_address, &address_length)
        );
        m_address = address_t{
            reinterpret_cast<unsigned char*>(worker_address),
            reinterpret_cast<unsigned char*>(worker_address) + address_length};
        ucp_worker_release_address(m_worker.get(), worker_address);
        m_worker.m_moved = false;
    }
                
    ucp_worker_h get() const noexcept { return m_worker.get(); }
    address_t address() const noexcept { return m_address; }
    void progress() const noexcept { ucp_worker_progress(m_worker.get()); }
};

} // namespace ucx
} // namespace tl
} // namespace ghex
} // namespace gridtools
