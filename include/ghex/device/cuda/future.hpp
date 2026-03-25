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

#include <ghex/config.hpp>
#include <ghex/device/cuda/event.hpp>
#include <ghex/device/cuda/error.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif
#include <memory>

#ifdef GHEX_CUDACC

namespace ghex
{
namespace device
{
/** @brief A future-like type that becomes ready once a cuda event is ready. The corresponding cuda stream
  * will be syncronized when waiting on this object. */
template<typename T>
struct future
{
    cuda_event m_event;
    T          m_data;

    future(T&& data, stream& stream)
    : m_event{}
    , m_data{std::move(data)}
    {
        GHEX_CHECK_CUDA_RESULT(cudaEventRecord(m_event.get(), stream));
    }

    future(const future&) = delete;
    future& operator=(const future&) = delete;
    future(future&& other) = default;
    future& operator=(future&&) = default;

    bool test() noexcept { return (m_event ? (cudaSuccess == cudaEventQuery(m_event.get())) : true); }

    void wait()
    {
        if (m_event) GHEX_CHECK_CUDA_RESULT(cudaEventSynchronize(m_event.get()));
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
    cuda_event m_event;

    future(stream& stream)
    : m_event{}
    {
        GHEX_CHECK_CUDA_RESULT(cudaEventRecord(m_event.get(), stream));
    }

    future(const future&) = delete;
    future& operator=(const future&) = delete;
    future(future&& other) = default;
    future& operator=(future&&) = default;

    bool test() noexcept { return (m_event ? (cudaSuccess == cudaEventQuery(m_event.get())) : true); }

    void wait()
    {
        if (m_event) GHEX_CHECK_CUDA_RESULT(cudaEventSynchronize(m_event.get()));
    }

    void get() { wait(); }
};

} // namespace device

} // namespace ghex

#endif
