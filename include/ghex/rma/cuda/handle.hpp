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

#include <ghex/device/cuda/error.hpp>
#include <ghex/rma/locality.hpp>
#include <ghex/rma/cuda/resource_cache.hpp>

namespace ghex
{
namespace rma
{
namespace cuda
{
struct info
{
    bool               m_on_gpu;
    void*              m_ptr;
    cudaIpcMemHandle_t m_cuda_handle;
};

struct local_data_holder
{
    bool               m_on_gpu;
    void*              m_ptr;
    cudaIpcMemHandle_t m_cuda_handle;

    local_data_holder(void* ptr, unsigned int, bool on_gpu)
    : m_on_gpu{on_gpu}
    , m_ptr{ptr}
    {
        // aquire the rma resource
        if (m_on_gpu) { GHEX_CHECK_CUDA_RESULT(cudaIpcGetMemHandle(&m_cuda_handle, ptr)); }
    }

    info get_info() const { return {m_on_gpu, m_ptr, m_cuda_handle}; }
};

struct remote_data_holder
{
    bool               m_on_gpu;
    locality           m_loc;
    int                m_rank;
    cudaIpcMemHandle_t m_cuda_handle;
    void*              m_cuda_ptr = nullptr;
    bool               m_attached = false;

    remote_data_holder(const info& info_, locality loc, int rank)
    : m_on_gpu{info_.m_on_gpu}
    , m_loc{loc}
    , m_rank{rank}
    , m_cuda_handle{info_.m_cuda_handle}
    {
        if (m_on_gpu && m_loc == locality::process)
        {
            auto&                              cache = get_cache();
            typename resource_cache::lock_type lk(cache.mtx());
            if (cache.find(m_rank) == cache.end()) attach(cache, info_.m_ptr);
            else if (cache[m_rank].find(info_.m_ptr) == cache[m_rank].end())
                attach(cache, info_.m_ptr);
            else
                m_cuda_ptr = cache[m_rank][info_.m_ptr];
        }
    }

    ~remote_data_holder()
    {
        // detach rma resource
        if (m_on_gpu && m_loc == locality::process && m_attached)
        { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaIpcCloseMemHandle(m_cuda_ptr)); }
    }

    void attach(resource_cache& cache, void* ptr)
    {
        // attach rma resource
        GHEX_CHECK_CUDA_RESULT(
            cudaIpcOpenMemHandle(&m_cuda_ptr, m_cuda_handle, cudaIpcMemLazyEnablePeerAccess));
        cache[m_rank][ptr] = m_cuda_ptr;
        m_attached = true;
    }

    void* get_ptr() const { return m_cuda_ptr; }
};

} // namespace cuda
} // namespace rma
} // namespace ghex
