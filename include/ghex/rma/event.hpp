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
#ifndef INCLUDED_GHEX_RMA_EVENT_HPP
#define INCLUDED_GHEX_RMA_EVENT_HPP

#include <memory>
#include "./range.hpp"

namespace gridtools {
namespace ghex {
namespace rma {
        
struct event_info
{
    bool m_source_on_gpu;
    bool m_target_on_gpu;
    locality m_locality;
#ifdef __CUDACC__
#endif
};

struct remote_event
{
    struct data_holder
    {
        bool m_source_on_gpu;
        bool m_target_on_gpu;
        locality m_locality;
#ifdef __CUDACC__
#endif
        data_holder(bool source_on_gpu, bool target_on_gpu, locality loc)
        : m_source_on_gpu{source_on_gpu}
        , m_target_on_gpu{target_on_gpu}
        , m_locality{loc}
#ifdef __CUDACC__
#endif
        {
        }

        event_info get_info() const
        {
            return { m_source_on_gpu, m_target_on_gpu, m_locality
#ifdef __CUDACC__
#endif
            };
        }

        void record()
        {
#ifdef __CUDACC__
            // if (source_on_gpu && m_locality == locality::thread) cudaEventRecord(...);
            // if (source_on_gpu && m_locality == locality::process) cudaEventRecord(...);
#endif
        }
    };

    std::unique_ptr<data_holder> m_impl;

    remote_event() = default;

    // create from source and deserialized target range
    remote_event(bool source_on_gpu, const range& r)
    : m_impl{std::make_unique<data_holder>(source_on_gpu, r.m_on_gpu, r.m_loc)}
    {}

    remote_event(remote_event&&) = default;
    remote_event& operator=(remote_event&&) = default;

    event_info get_info() const { return m_impl->get_info(); }

    void record()
    {
        m_impl->record();
    }
};

struct local_event
{
    struct data_holder
    {
        bool m_source_on_gpu;
        bool m_target_on_gpu;
        locality m_locality;
#ifdef __CUDACC__
#endif
        
        data_holder(const event_info& info_)
        : m_source_on_gpu{info_.m_source_on_gpu}
        , m_target_on_gpu{info_.m_target_on_gpu}
        , m_locality{info_.m_locality}
#ifdef __CUDACC__
#endif
        {
        }

        void wait()
        {
#ifdef __CUDACC__
            // if (source_on_gpu && m_locality == locality::thread) cudaEventSynchronize(m_cuda_thread_event);
            // if (source_on_gpu && m_locality == locality::process) cudaEventSynchronize(m_cuda_ipc_event);
#endif
        }
    };
        
    std::unique_ptr<data_holder> m_impl;

    local_event() = default;

    local_event(const event_info& info_)
    : m_impl{std::make_unique<data_holder>(info_)}
    {}
    
    local_event(local_event&&) = default;
    local_event& operator=(local_event&&) = default;

    void wait()
    {
        m_impl->wait();
    }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_EVENT_HPP */
