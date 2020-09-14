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

namespace gridtools {
namespace ghex {
namespace rma {
        
struct event_info
{
    bool m_target_on_gpu;
    locality m_locality;
#ifdef __CUDACC__
    cudaIpcEventHandle_t m_event_handle;
    cudaEvent_t m_event;
#endif
};

struct local_event
{
    struct data_holder
    {
        bool m_target_on_gpu;
        locality m_locality;
#ifdef __CUDACC__
        cudaEvent_t m_event;
        cudaIpcEventHandle_t m_event_handle;
#endif
        data_holder(bool target_on_gpu, locality loc)
        : m_target_on_gpu{target_on_gpu}
        , m_locality{loc}
        {
#ifdef __CUDACC__
            {
                if (m_loc == locality::thread)
                {
                    cudaEventCreate(&m_event, cudaEventDisableTiming);
                }
                if (m_loc == locality::process)
                {
                    cudaEventCreate(&m_event, cudaEventDisableTiming | cudaEventInterprocess);
                    cudaIpcGetEventHandle(&m_event_handle, m_event);
                }
            }
#endif
        }

        ~data_holder()
        {
            // destroy cuda resources
        }

        event_info get_info() const
        {
            return { m_target_on_gpu, m_locality
#ifdef __CUDACC__
                , m_event_handle, m_event
#endif
            };
        }

        void wait()
        {
#ifdef __CUDACC__
            cudaEventSynchronize(m_event);
#endif
        }
    };

    std::unique_ptr<data_holder> m_impl;

    local_event() = default;

    local_event(bool target_on_gpu, locality loc)
    : m_impl{std::make_unique<data_holder>(target_on_gpu, loc)}
    {}

    local_event(local_event&&) = default;
    local_event& operator=(local_event&&) = default;

    event_info get_info() const { return m_impl->get_info(); }

    void wait()
    {
        m_impl->wait();
    }
};

struct remote_event
{
    struct data_holder
    {
        bool m_source_on_gpu;
        bool m_target_on_gpu;
        locality m_locality;
#ifdef __CUDACC__
        cudaIpcEventHandle_t m_event_handle;
        cudaEvent_t m_event;
        cudaStream_t m_stream;
#endif
        
        data_holder(const event_info& info_, bool source_on_gpu)
        : m_source_on_gpu{source_on_gpu}
        , m_target_on_gpu{info_.m_target_on_gpu}
        , m_locality{info_.m_locality}
        {
#ifdef __CUDACC__
            if (m_source_on_gpu || m_target_on_gpu)
            {
                cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking);
                if (m_loc == locality::thread)
                {
                    m_event = info_.m_event;
                }
                if (m_loc == locality::process)
                {
                    m_event_handle = info_.m_event_handle;
                    cudaIpcOpenEventHandle(&m_event, m_event_handle);
                }
            }
#endif
        }

        void record()
        {
#ifdef __CUDACC__
            if (m_source_on_gpu || m_target_on_gpu) cudaEventRecord(m_event, m_stream);
#endif
        }
    };
        
    std::unique_ptr<data_holder> m_impl;

    remote_event() = default;

    remote_event(const event_info& info_, bool source_on_gpu)
    : m_impl{std::make_unique<data_holder>(info_, source_on_gpu)}
    {}
    
    remote_event(remote_event&&) = default;
    remote_event& operator=(remote_event&&) = default;

    void record()
    {
        m_impl->record();
    }

#ifdef __CUDACC__
    cudaStream_t get_stream() const { return m_impl->m_stream; }
#endif
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_EVENT_HPP */
