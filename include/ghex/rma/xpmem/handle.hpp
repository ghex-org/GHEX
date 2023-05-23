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

#include <ghex/rma/locality.hpp>

#include <cstdint>
#include <stdexcept>
extern "C"
{
#include <xpmem.h>
#include <unistd.h>
}

namespace ghex
{
namespace rma
{
namespace xpmem
{
// Below are implementations of a handle in a multi-process setting using xpmem.
// Please refer to the documentation in rma/handle.hpp for further explanations.

#define align_down_pow2(_n, _alignment) ((_n) & ~((_alignment)-1))

#define align_up_pow2(_n, _alignment) align_down_pow2((_n) + (_alignment)-1, _alignment)

struct info
{
    bool           m_on_gpu;
    std::uintptr_t m_xpmem_start;
    std::uintptr_t m_xpmem_size;
    std::uintptr_t m_xpmem_offset;
    xpmem_segid_t  m_xpmem_endpoint;
};

struct local_data_holder
{
    bool           m_on_gpu;
    std::uintptr_t m_page_size;
    std::uintptr_t m_xpmem_start;
    std::uintptr_t m_xpmem_end;
    std::uintptr_t m_xpmem_size;
    std::uintptr_t m_xpmem_offset;
    xpmem_segid_t  m_xpmem_endpoint;

    local_data_holder(void* ptr, unsigned int size, bool on_gpu)
    : m_on_gpu{on_gpu}
    {
        // aquire the rma resource
        if (!m_on_gpu)
        {
            m_page_size = getpagesize();
            m_xpmem_start = align_down_pow2((reinterpret_cast<std::uintptr_t>(ptr)), m_page_size);
            m_xpmem_end =
                align_up_pow2((reinterpret_cast<std::uintptr_t>(ptr) + size), m_page_size);
            m_xpmem_size = m_xpmem_end - m_xpmem_start;
            m_xpmem_offset = reinterpret_cast<std::uintptr_t>(ptr) - m_xpmem_start;
            m_xpmem_endpoint =
                xpmem_make((void*)m_xpmem_start, m_xpmem_size, XPMEM_PERMIT_MODE, (void*)0666);
            if (m_xpmem_endpoint < 0) throw std::runtime_error("could not register xpmem endpoint");
        }
    }

    ~local_data_holder()
    {
        if (!m_on_gpu) xpmem_remove(m_xpmem_endpoint);
    }

    info get_info() const
    {
        return {m_on_gpu, m_xpmem_start, m_xpmem_size, m_xpmem_offset, m_xpmem_endpoint};
    }
};

struct remote_data_holder
{
    bool           m_on_gpu;
    locality       m_loc;
    std::uintptr_t m_xpmem_start;
    std::uintptr_t m_xpmem_size;
    std::uintptr_t m_xpmem_offset;
    xpmem_segid_t  m_xpmem_endpoint;
    xpmem_addr     m_xpmem_addr;
    void*          m_xpmem_ptr = nullptr;

    remote_data_holder(const info& info_, locality loc, int)
    : m_on_gpu{info_.m_on_gpu}
    , m_loc{loc}
    , m_xpmem_start{info_.m_xpmem_start}
    , m_xpmem_size{info_.m_xpmem_size}
    , m_xpmem_offset{info_.m_xpmem_offset}
    , m_xpmem_endpoint{info_.m_xpmem_endpoint}
    {
        // attach rma resource
        if (!m_on_gpu && m_loc == locality::process)
        {
            m_xpmem_addr.offset = 0;
            m_xpmem_addr.apid = xpmem_get(m_xpmem_endpoint, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
            m_xpmem_ptr =
                (unsigned char*)xpmem_attach(m_xpmem_addr, m_xpmem_size, NULL) + m_xpmem_offset;
        }
    }

    ~remote_data_holder()
    {
        // detach rma resource
        if (!m_on_gpu && m_loc == locality::process)
        {
            xpmem_detach((void*)m_xpmem_start);
            xpmem_release(m_xpmem_addr.apid);
        }
    }

    void* get_ptr() const { return m_xpmem_ptr; }
};

} // namespace xpmem
} // namespace rma
} // namespace ghex
