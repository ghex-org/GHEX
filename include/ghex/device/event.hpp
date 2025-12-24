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

#if defined(GHEX_CUDACC)
#include <ghex/device/cuda/event.hpp>
#else
namespace ghex
{
namespace device
{
struct cuda_event
{
    cuda_event() {};
    cuda_event(const cuda_event&) = delete;
    cuda_event& operator=(const cuda_event&) = delete;
    cuda_event(cuda_event&& other) noexcept = default;
    cuda_event& operator=(cuda_event&&) noexcept = default;
    ~cuda_event() noexcept = default;
};

} // namespace device
} // namespace ghex
#endif
