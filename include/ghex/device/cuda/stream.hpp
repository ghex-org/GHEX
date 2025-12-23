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
#include <assert.h>
#include <memory>
#include <vector>

namespace ghex
{
namespace device
{
/** @brief thin wrapper around a cuda stream */
struct stream
{
    cudaStream_t          m_stream;
    ghex::util::moved_bit m_moved;

    stream(){GHEX_CHECK_CUDA_RESULT(cudaStreamCreateWithFlags(&m_stream, cudaStreamNonBlocking))}

    stream(const stream&) = delete;
    stream& operator=(const stream&) = delete;
    stream(stream&& other) noexcept = default;
    stream& operator=(stream&&) noexcept = default;

    ~stream()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaStreamDestroy(m_stream)) }
    }

    operator bool() const noexcept { return m_moved; }

    operator cudaStream_t() const noexcept
    {
        assert(!m_moved);
        return m_stream;
    }

    cudaStream_t& get() noexcept
    {
        assert(!m_moved);
        return m_stream;
    }
    const cudaStream_t& get() const noexcept
    {
        assert(!m_moved);
        return m_stream;
    }

    void sync()
    {
        // busy wait here
        assert(!m_moved);
        GHEX_CHECK_CUDA_RESULT(cudaStreamSynchronize(m_stream))
    }
};
} // namespace device
} // namespace ghex
