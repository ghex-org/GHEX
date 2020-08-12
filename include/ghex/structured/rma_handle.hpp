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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_HANDLE_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_HANDLE_HPP

#include "../arch_traits.hpp"
#include "../transport_layer/ri/types.hpp"
#ifdef GHEX_USE_XPMEM
#include "../transport_layer/ri/xpmem/data.hpp"
#endif /* GHEX_USE_XPMEM */

namespace gridtools {
namespace ghex {
namespace structured {
    
template<typename T, typename Arch, typename DomainDescriptor, int... Order>
class field_descriptor;

template<typename FieldDescriptor>
class rma_handle;

template<typename T, typename DomainDescriptor, int... Order>
class rma_handle<field_descriptor<T, cpu, DomainDescriptor, Order...>>
{
public:
    using derived = field_descriptor<T, cpu, DomainDescriptor, Order...>;
#ifdef GHEX_USE_XPMEM
    using rma_data_t = tl::ri::xpmem::data;
    std::shared_ptr<rma_data_t> m_xpmem_data_ptr;
#else
    using rma_data_t = int;
#endif /* GHEX_USE_XPMEM */

    derived* d_cast()
    {
        return static_cast<derived*>(this);
    }

    const derived* d_cast() const
    {
        return static_cast<const derived*>(this);
    }

    auto get_rma_data() const
    {
#ifdef GHEX_USE_XPMEM
        return *m_xmpmem_data_ptr;
#else
        return 0;
#endif
    }

    void reset_rma_data()
    {
#ifdef GHEX_USE_XPMEM
        new(&m_xpmem_data_ptr) std::shared_ptr<rma_data_t>{};
#endif
    }

    void init_rma_local()
    {
#ifdef GHEX_USE_XPMEM
        if (!m_xpmem_data_ptr)
        {
            auto size = d_cast()->m_extents[0];
            for (unsigned int i=1; i<d_cast()->m_extents.size(); ++i)
                size *= d_cast()->m_extents[i];
            m_xpmem_data_ptr = tl::ri::xpmem::make_local_data(d_cast()->m_data, size);
        }
#endif /* GHEX_USE_XPMEM */
    }

    void release_rma_local() { }

    void init_rma_remote(const rma_data_t& data, tl::ri::locality loc)
    {
#ifdef GHEX_USE_XPMEM
        if (loc == tl::ri::locality::process)
            if (!m_xpmem_data_ptr)
            {
                m_xpmem_data_ptr = tl::ri::xpmem::make_remote_data(data);
                d_cast()->m_data = (T*)m_xpmem_data_ptr->m_ptr;
            }
#endif /* GHEX_USE_XPMEM */	
    }
    
    void release_rma_remote() { }
};

template<typename T, typename DomainDescriptor, int... Order>
class rma_handle<field_descriptor<T, gpu, DomainDescriptor, Order...>>
{
public:
    using derived = field_descriptor<T, gpu, DomainDescriptor, Order...>;
#ifdef __CUDACC__
    using rma_data_t = cudaIpcMemHandle_t;
    std::shared_ptr<rma_data_t> m_cuda_data_ptr;
#else
    // used for emulated gpu fields
#ifdef GHEX_USE_XPMEM
    using rma_data_t = tl::ri::xpmem::data;
    std::shared_ptr<rma_data_t> m_xpmem_data_ptr;
#else
    using rma_data_t = int;
#endif
#endif

    derived* d_cast()
    {
        return static_cast<derived*>(this);
    }

    const derived* d_cast() const
    {
        return static_cast<const derived*>(this);
    }
    
    auto get_rma_data() const
    {
#ifdef __CUDACC__
        return *m_cuda_data_ptr;
#else
        // used for emulated gpu fields
#ifdef GHEX_USE_XPMEM
        return *m_xmpmem_data_ptr;
#else
        return 0;
#endif
#endif
    }

    void reset_rma_data()
    {
#ifdef __CUDACC__
        new(&m_cuda_data_ptr) std::shared_ptr<rma_data_t>{};
#else
        // used for emulated gpu fields
#ifdef GHEX_USE_XPMEM
        new(&m_xpmem_data_ptr) std::shared_ptr<rma_data_t>{};
#endif
#endif
    }

    void init_rma_local()
    {
#ifdef __CUDACC__
        if (!m_cuda_data_ptr)
        {
            auto h = new rma_data_t;
            cudaIpcGetMemHandle(h, d_cast()->m_data);
            m_cuda_data_ptr = std::shared_ptr<rma_data_t>{h};
        }
#else
        // used for emulated gpu fields
#ifdef GHEX_USE_XPMEM
        if (!m_xpmem_data_ptr)
        {
            auto size = d_cast()->m_extents[0];
            for (unsigned int i=1; i<d_cast()->m_extents.size(); ++i)
                size *= d_cast()->m_extents[i];
            m_xpmem_data_ptr = tl::ri::xpmem::make_local_data(d_cast()->m_data, size);
        }
#endif
#endif
    }

    void release_rma_local() { }

    void init_rma_remote(const rma_data_t& data, tl::ri::locality loc)
    {
#ifdef __CUDACC__
        if (loc == tl::ri::locality::process)
            if (!m_cuda_data_ptr)
            {
                void* vptr = (d_cast()->m_data);
                cudaIpcOpenMemHandle(&vptr, data, cudaIpcMemLazyEnablePeerAccess); 
                d_cast()->m_data = (T*)vptr;
                m_cuda_data_ptr = std::shared_ptr<rma_data_t>{
                    new rma_data_t{data},
                    [ptr = d_cast()->m_data](rma_data_t* h){
                        cudaIpcCloseMemHandle(ptr); 
                        delete h;
                    }
                };
            }
#else
        // used for emulated gpu fields
#ifdef GHEX_USE_XPMEM
        if (loc == tl::ri::locality::process)
            if (!m_xpmem_data_ptr)
            {
                m_xpmem_data_ptr = tl::ri::xpmem::make_remote_data(data);
                d_cast()->m_data = (T*)m_xpmem_data_ptr->m_ptr;
            }
#endif
#endif
    }
    
    void release_rma_remote() { }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_HANDLE_HPP */
