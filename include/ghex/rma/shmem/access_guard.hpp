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

#include <ghex/rma/access_mode.hpp>
#include <ghex/rma/locality.hpp>
#include <ghex/rma/shmem/handle.hpp>
#include <boost/interprocess/sync/interprocess_mutex.hpp>
#include <boost/interprocess/sync/interprocess_condition.hpp>
#include <boost/interprocess/sync/scoped_lock.hpp>
#include <memory>

namespace ghex
{
namespace rma
{
namespace shmem
{
// Below are implementations of access guards in a multi-process setting.
// Please refer to the documentation in rma/access_guard.hpp for further explanations.

struct access_state
{
    access_mode                                 m_mode = access_mode::local;
    boost::interprocess::interprocess_mutex     m_mtx;
    boost::interprocess::interprocess_condition m_cv;
};

struct local_access_guard
{
    struct impl
    {
        void*             m_ptr = nullptr;
        local_data_holder m_handle;
        access_state&     m_state;

        impl(access_mode m)
        : m_handle(&m_ptr, sizeof(access_state), false)
        , m_state{*(new (m_ptr) access_state{m, {}, {}})}
        {
        }
    };

    using lock_type = boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>;

    struct info
    {
        ghex::rma::shmem::info m_info;
    };

    std::unique_ptr<impl> m_impl;

    local_access_guard(access_mode m = access_mode::local)
    : m_impl{std::make_unique<impl>(m)}
    {
    }

    local_access_guard(local_access_guard&&) = default;

    info get_info() const { return {m_impl->m_handle.get_info()}; }

    void start_target_epoch()
    {
        lock_type lk{m_impl->m_state.m_mtx};
        m_impl->m_state.m_cv.wait(
            lk, [this] { return m_impl->m_state.m_mode == access_mode::local; });
    }

    bool try_start_target_epoch()
    {
        lock_type lk{m_impl->m_state.m_mtx};
        return m_impl->m_state.m_mode == access_mode::local;
    }

    void end_target_epoch()
    {
        {
            lock_type lk{m_impl->m_state.m_mtx};
            m_impl->m_state.m_mode = access_mode::remote;
        }
        m_impl->m_state.m_cv.notify_one();
    }
};

struct remote_access_guard
{
    using lock_type = boost::interprocess::scoped_lock<boost::interprocess::interprocess_mutex>;

    std::unique_ptr<remote_data_holder> m_handle;

    remote_access_guard(typename local_access_guard::info info_, locality loc, int rank)
    : m_handle{std::make_unique<remote_data_holder>(info_.m_info, loc, rank)}
    {
    }
    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;

    access_state* get_ptr() { return (access_state*)(m_handle->get_ptr()); }

    void start_source_epoch()
    {
        lock_type lk{get_ptr()->m_mtx};
        get_ptr()->m_cv.wait(lk, [this] { return get_ptr()->m_mode == access_mode::remote; });
    }

    bool try_start_source_epoch()
    {
        lock_type lk{get_ptr()->m_mtx};
        return get_ptr()->m_mode == access_mode::remote;
    }

    void end_source_epoch()
    {
        {
            lock_type lk{get_ptr()->m_mtx};
            get_ptr()->m_mode = access_mode::local;
        }
        get_ptr()->m_cv.notify_one();
    }
};

} // namespace shmem
} // namespace rma
} // namespace ghex
