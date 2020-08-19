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
#ifndef INCLUDED_GHEX_RMA_ACCESS_GUARD_HPP
#define INCLUDED_GHEX_RMA_ACCESS_GUARD_HPP

#include "./locality.hpp"
#include "./thread/access_guard.hpp"
#ifdef GHEX_USE_XPMEM
#include "./xpmem/access_guard.hpp"
#endif

namespace gridtools {
namespace ghex {
namespace rma {

struct local_access_guard
{
    locality m_locality;
    thread::local_access_guard m_thread_guard;
#ifdef GHEX_USE_XPMEM
    xpmem::local_access_guard m_process_guard;
#endif

    struct info
    {
        locality m_locality;
        thread::local_access_guard::info m_thread_guard_info;
#ifdef GHEX_USE_XPMEM
        xpmem::local_access_guard::info m_process_guard_info;
#endif
    };

    local_access_guard(locality loc)
    : m_locality{loc}
    {}

    local_access_guard(local_access_guard&&) = default;

    info get_info() const
    {
        return {m_locality
            , m_thread_guard.get_info()
#ifdef GHEX_USE_XPMEM
            , m_process_guard.get_info()
#endif
        };
    }

    locality get_locality() const { return m_locality; }
    
    void start_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_target_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard.start_target_epoch();
#endif
    }

    void end_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_target_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard.end_target_epoch();
#endif
    }
};

struct remote_access_guard
{
    locality m_locality;
    thread::remote_access_guard m_thread_guard;
#ifdef GHEX_USE_XPMEM
    xpmem::remote_access_guard m_process_guard;
#endif

    remote_access_guard(typename local_access_guard::info info_)
    : m_locality(info_.m_locality)
    , m_thread_guard(info_.m_thread_guard_info)
#ifdef GHEX_USE_XPMEM
    , m_process_guard(info_.m_process_guard_info)
#endif
    {}
    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;
    
    void start_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_source_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard.start_source_epoch();
#endif
    }
    
    void end_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_source_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard.end_source_epoch();
#endif
    }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_ACCESS_GUARD_HPP */
