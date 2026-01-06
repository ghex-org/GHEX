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
#include <cassert>
#include <memory>
#include <vector>

namespace ghex
{
namespace device
{
/** @brief thin wrapper around a cuda event */
struct cuda_event
{
    cudaEvent_t           m_event;
    ghex::util::moved_bit m_moved;

    cuda_event() {
        GHEX_CHECK_CUDA_RESULT(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming))
    };
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
    cuda_event(cuda_event&& other) noexcept = default;
    cuda_event& operator=(cuda_event&&) noexcept = default;

    ~cuda_event()
    {
        if (!m_moved) { GHEX_CHECK_CUDA_RESULT_NO_THROW(cudaEventDestroy(m_event)) }
    }

    //! Returns `true` is `*this` has been moved, i.e. is no longer a usable event.
    operator bool() const noexcept { return m_moved; }

    cudaEvent_t& get() noexcept
    {
        assert(!m_moved);
        return m_event;
    }
    const cudaEvent_t& get() const noexcept
    {
        assert(!m_moved);
        return m_event;
    }
};
} // namespace device
} // namespace ghex
