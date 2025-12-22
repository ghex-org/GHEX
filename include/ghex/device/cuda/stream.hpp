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
#include <ghex/device/cuda/error.hpp>
#include <ghex/device/cuda/runtime.hpp>
#include <ghex/util/moved_bit.hpp>
#include <memory>

namespace ghex
{
namespace device
{
struct cuda_event
{
    cudaEvent_t           m_event;
    ghex::util::moved_bit m_moved;

    cuda_event(){GHEX_CHECK_CUDA_RESULT(cudaEventCreateWithFlags(&m_event,
        cudaEventDisableTiming))} cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
    cuda_event(cuda_event&& other) = default;
    cuda_event& operator=(cuda_event&&) = default;

    ~cuda_event()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaEventDestroy(m_event)) }
    }

    operator bool() const noexcept { return m_moved; }
    operator cudaEvent_t() const noexcept { return m_event; }
    cudaEvent_t&       get() noexcept { return m_event; }
    const cudaEvent_t& get() const noexcept { return m_event; }
};

/** @brief thin wrapper around a cuda stream */
struct stream
{
    cudaStream_t          m_stream;
    ghex::util::moved_bit m_moved;

    stream(){GHEX_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking))}

    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;
    stream(stream&& other) = default;
    stream& operator=(stream&&) = default;

    ~stream()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaStreamDestroy(m_stream)) }
    }

    operator bool() const noexcept { return m_moved; }

    operator cudaStream_t() const noexcept { return m_stream; }

    cudaStream_t&       get() noexcept { return m_stream; }
    const cudaStream_t& get() const noexcept { return m_stream; }

    void sync()
    {
        // busy wait here
        GHEX_CHECK_CUDA_RESULT(cudaStreamSynchronize(m_stream))
    }
};
} // namespace device

} // namespace ghex
