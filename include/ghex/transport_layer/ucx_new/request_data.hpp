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
#include "./error.hpp"

namespace gridtools{
namespace ghex {
namespace tl {
namespace ucx {

enum class request_status : int { completed, in_progress };
enum class request_kind : int { invalid, send, recv };

struct request_header
{
    void* m_ucx_request;
    volatile request_status m_status = request_status::in_progress;

    request_header(std::size_t chunk_size) noexcept
    : m_ucx_request{(char*)this + chunk_size}
    {}

    request_header(const request_header&) = delete;
    request_header(request_header&&) = delete;

    void mark_completed() noexcept { m_status = request_status::completed; }

    bool ready() const noexcept { return m_status == request_status::completed; }
};

class request_data
{
public: // member types
    using pool_type = boost::pool<boost::default_user_allocator_malloc_free>;

private: // members
    pool_type* m_pool = nullptr;
    request_header* m_header = nullptr;
    bool m_detached = false;

public: // ctors
    request_data() noexcept {}

    request_data(pool_type* pool)
    : m_pool{pool}
    , m_header{new(check_malloc(m_pool->malloc())) request_header(m_pool->get_requested_size())}
    {}

    request_data(const request_data&) = delete;
    request_data& operator=(const request_data&) = delete;
    
    request_data(request_data&& other) noexcept
    : m_pool{std::exchange(other.m_pool, nullptr)}
    , m_header{std::exchange(other.m_header, nullptr)}
    , m_detached{std::exchange(other.m_detached, false)}
    {
    }

    request_data& operator=(request_data&& other) noexcept
    {
        m_pool = std::exchange(other.m_pool, nullptr);
        m_header = std::exchange(other.m_header, nullptr);
        m_detached = std::exchange(other.m_detached, false);
        return *this;
    }

    void swap(request_data& other) noexcept
    {
        std::swap(m_pool, other.m_pool);
        std::swap(m_header, other.m_header);
        std::swap(m_detached, other.m_detached);
    }

    ~request_data() { if (m_pool && !m_detached) m_pool->free(m_header); }

    void detach() noexcept
    {
        m_detached = true;
    }

public: // member functions
    request_header& get() const noexcept
    {
        assert((bool)m_pool);
        return *m_header;
    }

    bool ready() const noexcept
    {
        assert((bool)m_pool);
        return m_header->ready();
    }

    /*bool cancel()
    {
        return true;
    }*/

    bool valid() const noexcept { return (bool)m_pool; }

private: // private static member functions

    static void* check_malloc(void* ptr)
    {
        if (!ptr) throw std::bad_alloc();
        return ptr;
    }
};

class state;

class request
{
public: // member types
    using data_type = request_data;

private: // members
    data_type m_data;
    request_kind m_kind = request_kind::invalid;
    state* m_state = nullptr;

public: // ctors
    // invalid request by default
    request() noexcept {}

    request(data_type&& data, request_kind kind, state* st)
    : m_data{std::move(data)}
    , m_kind{kind}
    , m_state{st}
    {
        // preconditions
        assert(kind != request_kind::invalid);
        assert(m_data.valid());
    }

    request(const request&) = delete;
    request& operator=(const request&) = delete;
    
    request(request&& other) noexcept
    : m_data{std::move(other.m_data)}
    , m_kind{std::exchange(other.m_kind, request_kind::invalid)}
    , m_state{std::exchange(other.m_state, nullptr)}
    {
    }

    request& operator=(request&& other) noexcept
    {
        m_data = std::move(other.m_data);
        m_kind = std::exchange(other.m_kind, request_kind::invalid);
        m_state = std::exchange(other.m_state, nullptr);
        return *this;
    }

    void swap(request& other) noexcept
    {
        m_data.swap(other.m_data);
        std::swap(m_kind, other.m_kind);
        std::swap(m_state, other.m_state);
    }

    ~request()
    {
        // test whether request has been checked for completetion
        if (m_kind != request_kind::invalid)
        {
            // test whether it is ready
            if (!m_data.ready())
            {
                // detach lifetime (will be cleaned up at the end of the program)
                m_data.detach();
            }
        }
    }

public: // member functions
    bool test() noexcept
    {
        // invalid requests always return true
        if (m_kind == request_kind::invalid) return true;
        if (m_data.ready())
        {
            m_kind = request_kind::invalid;
            return true;
        }
        progress();
        if (m_data.ready())
        {
            m_kind = request_kind::invalid;
            return true;
        }
        return false;
    }

    void wait() noexcept
    {
        // invalid requests always return immediately
        if (m_kind == request_kind::invalid) return;
        while(!m_data.ready()) { progress(); }
        m_kind = request_kind::invalid;
        return;
    }
    
    bool ready() const noexcept
    {
        return m_data.ready();
    }
    
    bool cancel();

    data_type& data() noexcept { return m_data; }

private: // implementation
    void progress();
};

inline void swap(request& a, request& b) noexcept { a.swap(b); }

} // namespace ucx
} // namespace tl
} // namespace ghex
} // namespace gridtools
