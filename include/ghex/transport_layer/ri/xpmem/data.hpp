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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_XPMEM_DATA_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_XPMEM_DATA_HPP

#include <utility>
#include <memory>

extern "C"{
#include <xpmem.h>
#include <unistd.h>
}

#define align_down_pow2(_n, _alignment)		\
    ( (_n) & ~((_alignment) - 1) )

#define align_up_pow2(_n, _alignment)				\
    align_down_pow2((_n) + (_alignment) - 1, _alignment)

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {
namespace xpmem {

struct data
{
    xpmem_segid_t m_xpmem_endpoint = -1; ///< xpmem identifier for the m_data pointer
    size_t        m_xpmem_size = -1;     ///< size of the xpmem segment (page aligned)
    size_t        m_xpmem_offset = 0;    ///< offset to m_data within the page aligned xpmem segment
    xpmem_addr    m_xpmem_addr;
    void*         m_ptr = nullptr;
};

namespace detail {
template<typename T>
inline data init_local(T* ptr, size_t size)
{
    data m_data;
    m_data.m_ptr = ptr;
    size *= sizeof(T);
    // TODO: make general (GPUs, shmem)
    /* round the pointer to page boundaries, compute aligned size */
    int pagesize = getpagesize();
    uintptr_t start = align_down_pow2((uintptr_t)ptr, pagesize);
    uintptr_t end   = align_up_pow2((uintptr_t)((uintptr_t)ptr + size), pagesize);
    m_data.m_xpmem_size = end-start;
    m_data.m_xpmem_offset = (uintptr_t)ptr-start;

    /* publish pointer */
    m_data.m_xpmem_endpoint = xpmem_make((void*)start, m_data.m_xpmem_size, XPMEM_PERMIT_MODE, (void*)0666);
    if(m_data.m_xpmem_endpoint<0) fprintf(stderr, "error registering xpmem endpoint\n");
    return m_data;
}

inline void release_local(data& m_data)
{
    if (m_data.m_xpmem_endpoint >= 0)
    { 
        /*auto ret = */xpmem_remove(m_data.m_xpmem_endpoint);
        m_data.m_xpmem_endpoint = -1;
    }
}

struct local_deleter
{
    void operator()(data* data_ptr) const
    {
        release_local(*data_ptr);
        delete data_ptr;
    }
};

inline void init_remote(data& m_data)
{
    // TODO: make general (GPUs, shmem)
    m_data.m_xpmem_addr.offset = 0;
    m_data.m_xpmem_addr.apid   = xpmem_get(m_data.m_xpmem_endpoint, XPMEM_RDWR, XPMEM_PERMIT_MODE, NULL);
    m_data.m_ptr = (xpmem_attach(m_data.m_xpmem_addr, m_data.m_xpmem_size, NULL) + m_data.m_xpmem_offset);
}

inline void release_remote(data& m_data)
{
    if (m_data.m_xpmem_endpoint >= 0)
    { 
        int pagesize = getpagesize();
        uintptr_t start = align_down_pow2((uintptr_t)m_data.m_ptr, pagesize);
        /*auto ret = */xpmem_detach((void*)start);
        /*auto ret = */xpmem_release(m_data.m_xpmem_addr.apid);
        m_data.m_xpmem_endpoint = -1;
    }
}

struct remote_deleter
{
    void operator()(data* data_ptr) const
    {
        release_remote(*data_ptr);
        delete data_ptr;
    }
};

} // namespace detail

template<typename T>
inline std::shared_ptr<data> make_local_data(T* ptr, size_t size)
{
    return {new data{ detail::init_local(ptr,size) }, detail::local_deleter{} };
}

template<typename T>
inline std::shared_ptr<data> make_remote_data(data m_data)
{
    detail::init_remote(m_data);
    return {new data{ m_data }, detail::remote_deleter{} };
}

} // namespace xpmem
} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_XPMEM_DATA_HPP */
