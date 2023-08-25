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
#include <ghex/rma/xpmem/handle.hpp>
#include <memory>

namespace ghex
{
namespace rma
{
namespace xpmem
{
// Below are implementations of access guards in a multi-process setting.
// Please refer to the documentation in rma/access_guard.hpp for further explanations.

struct local_access_guard
{
    struct impl
    {
        struct mem
        {
            std::uintptr_t          m_page_size;
            unsigned char volatile* m_ptr;

            mem(access_mode m)
            : m_page_size{(unsigned)getpagesize()}
            {
                if (0 != posix_memalign((void**)&m_ptr, m_page_size, m_page_size))
                    throw std::runtime_error("cannot allocate xpmem access_guard\n");
                m_ptr[0] = static_cast<unsigned char>(m);
            }
        };

        mem               m_mem;
        local_data_holder m_handle;

        impl(access_mode m)
        : m_mem{m}
        , m_handle((void*)m_mem.m_ptr, m_mem.m_page_size, false)
        {
        }
    };

    struct info
    {
        ghex::rma::xpmem::info m_info;
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
        while (static_cast<unsigned char>(access_mode::local) != m_impl->m_mem.m_ptr[0])
        {
            // test if call to comm.progress() is beneficial for performance
            sched_yield();
        }
    }

    bool try_start_target_epoch()
    {
        return static_cast<unsigned char>(access_mode::local) == m_impl->m_mem.m_ptr[0];
    }

    void end_target_epoch()
    {
        m_impl->m_mem.m_ptr[0] = static_cast<unsigned char>(access_mode::remote);
    }
};

struct remote_access_guard
{
    std::unique_ptr<remote_data_holder> m_handle;

    remote_access_guard(typename local_access_guard::info info_, locality loc, int rank)
    : m_handle{std::make_unique<remote_data_holder>(info_.m_info, loc, rank)}
    {
    }

    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;

    unsigned char volatile* get_ptr() { return (unsigned char volatile*)(m_handle->get_ptr()); }

    void start_source_epoch()
    {
        while (static_cast<unsigned char>(access_mode::remote) != get_ptr()[0])
        {
            // TODO call comm.progress()
            sched_yield();
        }
    }

    bool try_start_source_epoch()
    {
        return static_cast<unsigned char>(access_mode::remote) == get_ptr()[0];
    }

    void end_source_epoch() { get_ptr()[0] = static_cast<unsigned char>(access_mode::local); }
};

} // namespace xpmem
} // namespace rma
} // namespace ghex
