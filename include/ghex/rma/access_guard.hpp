/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/rma/locality.hpp>
#include <ghex/rma/thread/access_guard.hpp>
#if defined(GHEX_USE_XPMEM_ACCESS_GUARD) && defined(GHEX_USE_XPMEM)
#include <ghex/rma/xpmem/access_guard.hpp>
#else
#include <ghex/rma/shmem/access_guard.hpp>
#endif

namespace ghex
{
namespace rma
{
/** @brief General local access guard wich synchronizes between the two participants in a RMA put
  * operation. This object is created at the site of the owner. All essential information can be
  * extracted through get_info(). The returned info object is POD and can be sent through the network
  * to the remote counter part.
  *
  * Access guards model a finite state machine. The only state is called epoch and defines who has
  * read/write access to a resource. The local access guard below can
  * - start a target epoch: busy wait until the resource has been freed by the remote counterpart
  * - end a target epoch: signal the end of read/write access to the remote counterpart
  * */
struct local_access_guard
{
    locality                   m_locality;
    thread::local_access_guard m_thread_guard;
#if defined(GHEX_USE_XPMEM_ACCESS_GUARD) && defined(GHEX_USE_XPMEM)
    using process_guard_type = xpmem::local_access_guard;
#else
    using process_guard_type = shmem::local_access_guard;
#endif
    process_guard_type m_process_guard;

    struct info
    {
        locality                          m_locality;
        thread::local_access_guard::info  m_thread_guard_info;
        typename process_guard_type::info m_process_guard_info;
    };

    local_access_guard(locality loc, access_mode m = access_mode::local)
    : m_locality{loc}
    , m_thread_guard(m)
    , m_process_guard(m)
    {
    }

    local_access_guard(local_access_guard&&) = default;

    info get_info() const
    {
        return {m_locality, m_thread_guard.get_info(), m_process_guard.get_info()};
    }

    locality get_locality() const { return m_locality; }

    void start_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_target_epoch();
        if (m_locality == locality::process) m_process_guard.start_target_epoch();
    }

    bool try_start_target_epoch()
    {
        if (m_locality == locality::thread) return m_thread_guard.try_start_target_epoch();
        if (m_locality == locality::process) return m_process_guard.try_start_target_epoch();
        return true;
    }

    void end_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_target_epoch();
        if (m_locality == locality::process) m_process_guard.end_target_epoch();
    }
};

/** @brief General remote access guard wich synchronizes between the two participants in a RMA put
  * operation. This object is created at the site of the remote and is constructed from an info
  * object obtained from the local counter part.
  *
  * Access guards model a finite state machine. The only state is called epoch and defines who has
  * read/write access to a resource. The remote access guard below can
  * - start a source epoch: busy wait until the resource has been freed by the resource owning process
  * - end a source epoch: signal the end of read/write access to the resource owning process
  * */
struct remote_access_guard
{
    locality                    m_locality;
    thread::remote_access_guard m_thread_guard;
#if defined(GHEX_USE_XPMEM_ACCESS_GUARD) && defined(GHEX_USE_XPMEM)
    xpmem::remote_access_guard m_process_guard;
#else
    shmem::remote_access_guard m_process_guard;
#endif

    remote_access_guard(typename local_access_guard::info info_, int rank)
    : m_locality(info_.m_locality)
    , m_thread_guard(info_.m_thread_guard_info, m_locality, rank)
    , m_process_guard(info_.m_process_guard_info, m_locality, rank)
    {
    }

    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;

    void start_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_source_epoch();
        if (m_locality == locality::process) m_process_guard.start_source_epoch();
    }

    bool try_start_source_epoch()
    {
        if (m_locality == locality::thread) return m_thread_guard.try_start_source_epoch();
        if (m_locality == locality::process) return m_process_guard.try_start_source_epoch();
        return true;
    }

    void end_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_source_epoch();
        if (m_locality == locality::process) m_process_guard.end_source_epoch();
    }
};

} // namespace rma
} // namespace ghex
