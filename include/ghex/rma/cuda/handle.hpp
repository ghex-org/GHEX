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
#ifndef INCLUDED_GHEX_RMA_CUDA_HANDLE_HPP
#define INCLUDED_GHEX_RMA_CUDA_HANDLE_HPP

#include "../locality.hpp"


namespace gridtools {
namespace ghex {
namespace rma {
namespace cuda {

struct info
{
    bool m_on_gpu;
    cudaIpcMemHandle_t m_cuda_handle;
};

struct local_data_holder
{
    bool m_on_gpu;
    cudaIpcMemHandle_t m_cuda_handle;
    
    local_data_holder(void* ptr, unsigned int, bool on_gpu)
    : m_on_gpu{on_gpu}
    {
        // aquire the rma resource
        if (m_on_gpu)
        {
            cudaIpcGetMemHandle(&m_cuda_handle, ptr);
        }
    }
    
    info get_info() const
    {
        return {m_on_gpu, m_cuda_handle};
    }
};

struct remote_data_holder
{
    bool m_on_gpu;
    cudaIpcMemHandle_t m_cuda_handle;
    void* m_cuda_ptr;
        
    remote_data_holder(const info& info_)
    : m_on_gpu{info_.m_on_gpu}
    , m_cuda_handle{info_.m_cuda_handle}
    {
        // attach rma resource
        if (m_on_gpu)
        {
            cudaIpcOpenMemHandle(&m_cuda_ptr, m_cuda_handle, cudaIpcMemLazyEnablePeerAccess);
        }
    }

    ~remote_data_holder()
    {
        // detach rma resource
        if (m_on_gpu)
        {
            cudaIpcCloseMemHandle(m_cuda_ptr); 
        }
    }

    void* get_ptr() const
    {
        return m_cuda_ptr;
    }
};

} // namespace cuda
} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_CUDA_HANDLE_HPP */
