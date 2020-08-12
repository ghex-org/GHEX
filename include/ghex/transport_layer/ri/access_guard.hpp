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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_ACCESS_GUARD_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_ACCESS_GUARD_HPP

#include "./thread/access_guard.hpp"
#ifdef GHEX_USE_XPMEM
#include "./xpmem/access_guard.hpp"
#endif

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {


struct access_guard
{
    thread::access_guard m_thread_guard;
#ifdef GHEX_USE_XPMEM
    xpmem::access_guard m_process_guard;
#endif

    access_guard() = default;
    access_guard(access_guard&& other) = default;

};

struct access_guard_view
{
    locality m_locality = locality::thread;
    thread::access_guard_view m_thread_guard_view;
#ifdef GHEX_USE_XPMEM
    xpmem::access_guard_view m_process_guard_view;
#endif

    access_guard_view() = default;

    access_guard_view(access_guard& g, locality loc)
    : m_locality{loc}
    , m_thread_guard_view{ g.m_thread_guard }
#ifdef GHEX_USE_XPMEM
    , m_process_guard_view{ g.m_process_guard }
#endif
    {}

    locality get_locality() const noexcept { return m_locality; }

    void init_remote()
    {
        if (m_locality == locality::thread) m_thread_guard_view.init_remote();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.init_remote();
#endif
    }

    void release_remote()
    {
        if (m_locality == locality::thread) m_thread_guard_view.release_remote();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.release_remote();
#endif
    }

    void start_remote_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard_view.start_remote_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.start_remote_epoch();
#endif
    }

    void end_remote_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard_view.end_remote_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.end_remote_epoch();
#endif
    }

    void start_local_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard_view.start_local_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.start_local_epoch();
#endif
    }

    void end_local_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard_view.end_local_epoch();
#ifdef GHEX_USE_XPMEM
        if (m_locality == locality::process) m_process_guard_view.end_local_epoch();
#endif
    }
    
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_ACCESS_GUARD_HPP */
