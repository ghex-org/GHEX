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
#ifndef INCLUDED_GHEX_RMA_XPMEM_ACCESS_GUARD2_HPP
#define INCLUDED_GHEX_RMA_XPMEM_ACCESS_GUARD2_HPP

#include <memory>
#include "../access_mode.hpp"
#include "./handle.hpp"

namespace gridtools {
namespace ghex {
namespace rma {
namespace xpmem {

struct local_access_guard
{
    struct impl
    {
        struct mem
        {
            std::uintptr_t m_page_size;
            unsigned char volatile* m_ptr;

            mem()
            : m_page_size{getpagesize()}
            {
                if(0 != posix_memalign((void**)&m_ptr, m_page_size, m_page_size))
                    throw std::runtime_error("cannot allocate xpmem access_guard\n");
                m_ptr[0] = static_cast<unsigned char>(access_mode::local);
            }
        };
    
        mem m_mem;
        local_data_holder m_handle;
        
        impl()
        : m_mem{}
        , m_handle(m_mem.m_ptr, m_mem.m_page_size, false)    
        {}
    };

    struct info
    {
        ::gridtools::ghex::rma::xpmem::info m_info;
    };

    std::unique_ptr<impl> m_impl;

    local_access_guard()
    : m_impl{std::make_unique<impl>()}
    {}
    
    local_access_guard(local_access_guard&&) = default;

    info get_info() const
    {
        return { m_impl->m_handle.get_info()  };
    }

    void start_target_epoch()
    {
        while(static_cast<unsigned char>(access_mode::local) != m_impl->m_mem.m_ptr[0])
        {
            // TODO call comm.progress()
            sched_yield();
        }
    }
    
    void end_target_epoch()
    {
        m_impl->m_mem.m_ptr[0] = static_cast<unsigned char>(access_mode::remote);
    }
};


struct remote_access_guard
{
    std::unique_ptr<remote_data_holder> m_handle;

    remote_access_guard(typename local_access_guard::info info_)
    : m_handle{std::make_unique<remote_data_holder>(info_.m_info)}
    {}
    remote_access_guard() = default;
    remote_access_guard(remote_access_guard&&) = default;
    remote_access_guard& operator=(remote_access_guard&&) = default;

    unsigned char volatile* get_ptr()
    {
        return (unsigned char volatile*)(m_handle->get_ptr());
    } 

    void start_source_epoch()
    {
        while(static_cast<unsigned char>(access_mode::remote) != get_ptr()[0])
        {
            // TODO call comm.progress()
            sched_yield();
        }
    }

    void end_source_epoch()
    {
        get_ptr()[0] = static_cast<unsigned char>(access_mode::local);
    }
};

} // namespace xpmem
} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_XPMEM_ACCESS_GUARD2_HPP */
