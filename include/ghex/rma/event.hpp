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
#include "../cuda_utils/error.hpp"

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC
#include "../common/cuda_runtime.hpp"
#endif

namespace gridtools {
namespace ghex {
namespace rma {

// POD event data
struct event_info
{
    bool m_target_on_gpu;
    locality m_loc;
#ifdef GHEX_CUDACC
    cudaIpcEventHandle_t m_event_handle;
    cudaEvent_t m_event;
#endif
};

/** @brief Events are used to synchronize cuda streams across threads/processes. Similar to handles,
  * they consist of a local and a remote part and are associated with a range.
  *
  * A local event is created at the put target site and can be waited upon. The asynchronous GPU
  * kernels involved with the one-sided put are guaranteed to be finished once the wait function
  * returns. */
struct local_event
{
    struct data_holder
    {
        bool m_target_on_gpu;
        locality m_loc;
#ifdef GHEX_CUDACC
        cudaEvent_t m_event;
        cudaIpcEventHandle_t m_event_handle;
#endif
        data_holder(bool target_on_gpu, locality loc)
        : m_target_on_gpu{target_on_gpu}
        , m_loc{loc}
        {
#ifdef GHEX_CUDACC
            if (m_loc == locality::thread)
            {
                GHEX_CHECK_CUDA_RESULT(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
            }
            if (m_loc == locality::process)
            {
                GHEX_CHECK_CUDA_RESULT(
                    cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming | cudaEventInterprocess));
                GHEX_CHECK_CUDA_RESULT(cudaIpcGetEventHandle(&m_event_handle, m_event));
            }
#endif
        }

        ~data_holder()
        {
#ifdef GHEX_CUDACC
            if (m_loc != locality::remote)
            {
                cudaEventDestroy(m_event);
            }
#endif
        }

        event_info get_info() const
        {
            return { m_target_on_gpu, m_loc
#ifdef GHEX_CUDACC
                , m_event_handle, m_event
#endif
            };
        }

        void wait()
        {
#ifdef GHEX_CUDACC
            GHEX_CHECK_CUDA_RESULT(cudaEventSynchronize(m_event));
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

/** @brief Events are used to synchronize cuda streams across threads/processes. Similar to handles,
  * they consist of a local and a remote part and are associated with a range.
  *
  * A remote event is created at the put source site. It exposes a stream and it allows for
  * registering an event. This event can then be waited upon by the local counterpart. All
  * asynchronous GPU kernels should use the provided stream. */
struct remote_event
{
    struct data_holder
    {
        bool m_source_on_gpu;
        bool m_target_on_gpu;
        locality m_loc;
#ifdef GHEX_CUDACC
        cudaIpcEventHandle_t m_event_handle;
        cudaEvent_t m_event;
        cudaStream_t m_stream;
#endif
        
        data_holder(const event_info& info_, bool source_on_gpu)
        : m_source_on_gpu{source_on_gpu}
        , m_target_on_gpu{info_.m_target_on_gpu}
        , m_loc{info_.m_loc}
        {
#ifdef GHEX_CUDACC
            if (m_source_on_gpu || m_target_on_gpu)
            {
                GHEX_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking));
                if (m_loc == locality::thread)
                {
                    m_event = info_.m_event;
                }
                if (m_loc == locality::process)
                {
                    m_event_handle = info_.m_event_handle;
                    GHEX_CHECK_CUDA_RESULT(cudaIpcOpenEventHandle(&m_event, m_event_handle));
                }
            }
#endif
        }

        ~data_holder()
        {
#ifdef GHEX_CUDACC
            if (m_source_on_gpu || m_target_on_gpu)
                cudaStreamDestroy(m_stream);
#endif
        }

        void record()
        {
#ifdef GHEX_CUDACC
            if (m_source_on_gpu || m_target_on_gpu)
                GHEX_CHECK_CUDA_RESULT(cudaEventRecord(m_event, m_stream));
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

#ifdef GHEX_CUDACC
    cudaStream_t get_stream() const { return m_impl->m_stream; }
#endif
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_EVENT_HPP */
