/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_CUDA_FUTURE_HPP
#define INCLUDED_CUDA_FUTURE_HPP

#include "../common/c_managed_struct.hpp"
#include "./error.hpp"
#include <memory>

#ifdef __CUDACC__

namespace gridtools {
    namespace ghex {
        namespace cuda {

            template<typename T>
            struct future
            {
                GHEX_C_MANAGED_STRUCT(event_type, cudaEvent_t, cudaEventCreateWithFlags, cudaEventDestroy)

                event_type m_event;
                T m_data;

                future(T&& data, ::gridtools::ghex::cuda_stream& stream)
                : m_event{cudaEventDisableTiming}
                //: m_event{cudaEventDisableTiming | cudaEventBlockingSync}
                , m_data{std::move(data)}
                {
                    GHEX_CHECK_CUDA_RESULT( cudaEventRecord(m_event, *stream.get()) );
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

                future(::gridtools::ghex::cuda_stream& stream)
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

            //template<typename T>
            //struct future
            //{
            //    struct event_deleter
            //    {
            //        void operator()(cudaEvent_t* s_ptr)
            //        {
            //            cudaEventDestroy(*s_ptr);
            //            delete s_ptr;
            //        }
            //    };
            //    T m_data;
            //    //cudaEvent_t m_event;
            //    //bool m_valid = true;
            //    std::unique_ptr<cudaEvent_t,event_deleter> m_event;

            //    future(T&& data, ::gridtools::ghex::cuda_stream& stream)
            //    : m_data{std::move(data)}
            //    , m_event{new cudaEvent_t}
            //    {
            //        GHEX_CHECK_CUDA_RESULT(
            //            cudaEventCreateWithFlags(m_event.get(), cudaEventDisableTiming) 
            //            //cudaEventCreateWithFlags(&m_event, cudaEventBlockingSync | cudaEventDisableTiming) 
            //        );
            //        GHEX_CHECK_CUDA_RESULT(
            //            cudaEventRecord(*m_event, *stream.get()) 
            //        );
            //    }
            //
            //    future(const future&) = delete;
            //    future& operator=(const future&) = delete;

            //    future(future&& other)=default;
            //    /*: m_data{std::move(other.m_data)}
            //    , m_event{std::move(other.m_event)}
            //    {
            //        other.m_valid = false;
            //    } */

            //    future& operator=(future&&) = default; /* other)
            //    {
            //        if (m_valid)
            //            GHEX_CHECK_CUDA_RESULT(
            //                cudaEventDestroy(m_event)
            //            );
            //        auto data_ptr = &m_data;
            //        m_data.~T();
            //        ::new((void*)data_ptr) T(std::move(other.m_data));
            //        auto event_ptr = &m_event;
            //        m_event.~cudaEvent_t();
            //        ::new((void*)event_ptr) cudaEvent_t(std::move(other.m_event));
            //        other.m_valid = false;
            //    }*/

            //    /*~future()
            //    {
            //        if (m_valid)
            //            //GHEX_CHECK_CUDA_RESULT(
            //                cudaEventDestroy(*m_event);
            //            //); 
            //    }*/

            //    bool test()
            //    {
            //        //return (m_valid ? (cudaSuccess == cudaEventQuery(m_event)) : true); 
            //        return (m_event ? (cudaSuccess == cudaEventQuery(*m_event)) : true); 
            //    }

            //    void wait()
            //    {
            //        //if (m_valid)
            //        if (m_event)
            //            GHEX_CHECK_CUDA_RESULT(
            //                cudaEventSynchronize(*m_event)
            //            );
            //    }

            //    [[nodiscard]] T get()
            //    {
            //        wait();
            //        // undefined behaviour if not valid future
            //        // if (!m_valid) throw std::runtime_error();
            //        return std::move(m_data);
            //    }

            //};

            //template<>
            //struct future<void>
            //{
            //    cudaEvent_t m_event;
            //
            //    future(const future&) = delete;
            //    future(future&&) = default;
            //    future& operator=(const future&) = delete;
            //    future& operator=(future&&) = default;

            //    future(::gridtools::ghex::cuda_stream& stream)
            //    {
            //        GHEX_CHECK_CUDA_RESULT(
            //            cudaEventCreateWithFlags (&m_event, /*cudaEventBlockingSync |*/ cudaEventDisableTiming) 
            //        );
            //        GHEX_CHECK_CUDA_RESULT(
            //            cudaEventRecord(m_event, *stream.get()) 
            //        );
            //    }

            //    bool test()
            //    {
            //        return cudaSuccess == cudaEventQuery(m_event); 
            //    }

            //    void wait()
            //    {
            //        GHEX_CHECK_CUDA_RESULT(
            //            cudaEventSynchronize(m_event)
            //        );
            //    }

            //    void get()
            //    {
            //        wait();
            //    }

            //};
        }
    }
}

#endif

#endif // INCLUDED_CUDA_FUTURE_HPP

