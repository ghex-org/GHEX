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
#ifndef INCLUDED_GHEX_CUDA_FUTURE_HPP
#define INCLUDED_GHEX_CUDA_FUTURE_HPP

#include "../common/c_managed_struct.hpp"
#include "./error.hpp"
#include <memory>

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC
#include "../common/cuda_runtime.hpp"
#endif

#ifdef GHEX_CUDACC

namespace gridtools {

    namespace ghex {

        namespace cuda {

            /** @brief A future-like type that becomes ready once a cuda event is ready. The corresponding cuda stream
              * will be syncronized when waiting on this object. */
            template<typename T>
            struct future
            {
                GHEX_C_MANAGED_STRUCT(event_type, cudaEvent_t, cudaEventCreateWithFlags, cudaEventDestroy)

                event_type m_event;
                T m_data;

                future(T&& data, stream& stream)
                : m_event{cudaEventDisableTiming}
                //: m_event{cudaEventDisableTiming | cudaEventBlockingSync}
                , m_data{std::move(data)}
                {
                    GHEX_CHECK_CUDA_RESULT( cudaEventRecord(m_event, stream) );
                }

                future(const future&) = delete;
                future& operator=(const future&) = delete;
                future(future&& other) = default;
                future& operator=(future&&) = default;

                bool test() noexcept
                {
                    return (m_event ? (cudaSuccess == cudaEventQuery(m_event)) : true); 
                }

                void wait()
                {
                    if (m_event)
                        GHEX_CHECK_CUDA_RESULT( cudaEventSynchronize(m_event) );
                }

                [[nodiscard]] T get()
                {
                    wait();
                    return std::move(m_data);
                }
            };

            template<>
            struct future<void>
            {
                GHEX_C_MANAGED_STRUCT(event_type, cudaEvent_t, cudaEventCreateWithFlags, cudaEventDestroy)

                event_type m_event;

                future(stream& stream)
                : m_event{cudaEventDisableTiming}
                //: m_event{cudaEventDisableTiming | cudaEventBlockingSync}
                {
                    GHEX_CHECK_CUDA_RESULT( cudaEventRecord(m_event, stream) );
                }

                future(const future&) = delete;
                future& operator=(const future&) = delete;
                future(future&& other) = default;
                future& operator=(future&&) = default;

                bool test() noexcept
                {
                    return (m_event ? (cudaSuccess == cudaEventQuery(m_event)) : true); 
                }

                void wait()
                {
                    if (m_event)
                        GHEX_CHECK_CUDA_RESULT( cudaEventSynchronize(m_event) );
                }

                void get()
                {
                    wait();
                }
            };

        } // namespace cuda

    } // namespace ghex

} // namespace gridtools

#endif

#endif /* INCLUDED_GHEX_CUDA_FUTURE_HPP */

