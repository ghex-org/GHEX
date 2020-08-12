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
    std::shared_ptr<rma_data_t> m_rma_data;
#else
    using rma_data_t = int;
    rma_data_t m_rma_data = 0;
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
        return *m_rma_data;
#else
        return m_rma_data;
#endif
    }

    void reset_rma_data()
    {
#ifdef GHEX_USE_XPMEM
        new(&m_rma_data) std::shared_ptr<rma_data_t>{};
#endif
    }

    void init_rma_local()
    {
#ifdef GHEX_USE_XPMEM
        if (!m_rma_data)
        {
            auto size = d_cast()->m_extents[0];
            for (unsigned int i=1; i<d_cast()->m_extents.size(); ++i) size *= d_cast()->m_extents[i];
            m_rma_data = tl::ri::xpmem::make_local_data(d_cast()->m_data, size);
        }
#endif /* GHEX_USE_XPMEM */
    }

    void release_rma_local() { }

    void init_rma_remote(const rma_data_t& data)
    {
#ifdef GHEX_USE_XPMEM
        if (!m_rma_data)
        {
            m_rma_data = tl::ri::xpmem::make_remote_data(data);
            d_cast()->m_data = (T*)m_rma_data->m_ptr;
        }
#else
        m_rma_data = data;
#endif /* GHEX_USE_XPMEM */	
    }
    
    void release_rma_remote() { }
};

template<typename T, typename DomainDescriptor, int... Order>
class rma_handle<field_descriptor<T, gpu, DomainDescriptor, Order...>>
{
public:
    using derived = field_descriptor<T, gpu, DomainDescriptor, Order...>;

#ifdef GHEX_USE_XPMEM
#ifdef __CUDACC__
    using rma_data_t = cudaIpcMemHandle_t;
    std::shared_ptr<rma_data_t> m_rma_data;
#else
    using rma_data_t = int;
    rma_data_t m_rma_data = 0;
#endif /* __CUDACC__ */
#else
    using rma_data_t = int;
    rma_data_t m_rma_data = 0;
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
#ifdef __CUDACC__
        return *m_rma_data;
#else
        return m_rma_data;
#endif /* __CUDACC__ */
#else
        return m_rma_data;
#endif /* GHEX_USE_XPMEM */	
    }

    void reset_rma_data()
    {
#ifdef GHEX_USE_XPMEM
#ifdef __CUDACC__
        new(&m_rma_data) std::shared_ptr<rma_data_t>{};
#endif /* __CUDACC__ */
#endif
    }

    void init_rma_local()
    {
#ifdef GHEX_USE_XPMEM
#ifdef __CUDACC__
        if (!m_rma_data)
        {
            auto h = new rma_data_t;
            cudaIpcGetMemHandle(h, d_cast()->m_data);
            m_rma_data = std::shared_ptr<rma_data_t>{h};
        }
#endif /* __CUDACC__ */
#endif /* GHEX_USE_XPMEM */	
    }

    void release_rma_local() { }

    void init_rma_remote(const rma_data_t& data)
    {
#ifdef GHEX_USE_XPMEM
#ifdef __CUDACC__
        if (!m_rma_data)
        {
            void* vptr = (d_cast()->m_data);
            cudaIpcOpenMemHandle(&vptr, data, cudaIpcMemLazyEnablePeerAccess); 
            d_cast()->m_data = (T*)vptr;
            m_rma_data = std::shared_ptr<rma_data_t>{
                new rma_data_t{data},
                [ptr = d_cast()->m_data](rma_data_t* h){
                    cudaIpcCloseMemHandle(ptr); 
                    delete h;
                }
            };
        }
#else
        m_rma_data = data;
#endif /* __CUDACC__ */
#else
        m_rma_data = data;
#endif /* GHEX_USE_XPMEM */	
    }
    
    void release_rma_remote() { }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_HANDLE_HPP */
