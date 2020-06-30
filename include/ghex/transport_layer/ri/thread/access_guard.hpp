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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_THREAD_ACCESS_GUARD_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_THREAD_ACCESS_GUARD_HPP

#include <mutex>
#include <condition_variable>
#include <memory>

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {
namespace thread {

// a finite-state machine that guards alternating either the local or remote site
// must be initialized with the init function
// initialization completion can be checked with ready function
// for now this is only for threads, but should be made more general for processes (and
// remote processes)
struct access_guard
{
    enum access_mode
    {
        local,
        remote
    };

    struct impl
    {
        access_mode             mode = local;
        std::mutex              mtx;
        std::condition_variable cv;
    };

    std::unique_ptr<impl> m_impl;

    access_guard() : m_impl{std::make_unique<impl>()} {}
};

// a view on an access guard
// does not own any resources
// exposes necessary functions to lock and unlock remote/local sites
struct access_guard_view
{
    access_guard::impl* m_impl = nullptr;

    access_guard_view() = default;

    access_guard_view(access_guard& g) : m_impl{g.m_impl.get()} {}

    void init_remote() {}
    void release_remote() {}

    void start_remote_epoch() {
        std::unique_lock<std::mutex> lk{m_impl->mtx};
        m_impl->cv.wait(lk, [this] { return m_impl->mode == access_guard::remote; });
    }

    void end_remote_epoch() {
        {
            std::lock_guard<std::mutex> lk{m_impl->mtx};
            m_impl->mode = access_guard::local;
        }
        m_impl->cv.notify_one();
    }

    void start_local_epoch() {
        std::unique_lock<std::mutex> lk{m_impl->mtx};
        m_impl->cv.wait(lk, [this] { return m_impl->mode == access_guard::local; });
    }

    void end_local_epoch() {
        {
            std::lock_guard<std::mutex> lk{m_impl->mtx};
            m_impl->mode = access_guard::remote;
        }
        m_impl->cv.notify_one();
    }
};

} // namespace thread
} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_THREAD_ACCESS_GUARD_HPP */
