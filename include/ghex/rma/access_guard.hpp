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
#include "./shmem/access_guard.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

/** @brief General local access guard wich synchronizes between the two participants in a RMA put
 * operation. This object is created at the site of the owner. All essential information can be
 * extracted through get_info(). The returned info object is POD and can be sent through the network
 * to the remote counter part. */
struct local_access_guard
{
    locality m_locality;
    thread::local_access_guard m_thread_guard;
    shmem::local_access_guard m_process_guard;

    struct info
    {
        locality m_locality;
        thread::local_access_guard::info m_thread_guard_info;
        shmem::local_access_guard::info m_process_guard_info;
    };

    local_access_guard(locality loc)
    : m_locality{loc}
    {}

    local_access_guard(local_access_guard&&) = default;

    info get_info() const
    {
        return {m_locality
            , m_thread_guard.get_info()
            , m_process_guard.get_info()
        };
    }

    locality get_locality() const { return m_locality; }
    
    void start_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_target_epoch();
        if (m_locality == locality::process) m_process_guard.start_target_epoch();
    }

    void end_target_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_target_epoch();
        if (m_locality == locality::process) m_process_guard.end_target_epoch();
    }
};

/** @brief General remote access guard wich synchronizes between the two participants in a RMA put
 * operation. This object is created at the site of the remote and is constructed from an info
 * object obtained from the local counter part. */
struct remote_access_guard
{
    locality m_locality;
    thread::remote_access_guard m_thread_guard;
    shmem::remote_access_guard m_process_guard;

    remote_access_guard(typename local_access_guard::info info_)
    : m_locality(info_.m_locality)
    , m_thread_guard(info_.m_thread_guard_info, m_locality)
    , m_process_guard(info_.m_process_guard_info, m_locality)
    {}

    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;
    
    void start_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.start_source_epoch();
        if (m_locality == locality::process) m_process_guard.start_source_epoch();
    }
    
    void end_source_epoch()
    {
        if (m_locality == locality::thread) m_thread_guard.end_source_epoch();
        if (m_locality == locality::process) m_process_guard.end_source_epoch();
    }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_ACCESS_GUARD_HPP */
