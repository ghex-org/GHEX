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
#include <ghex/device/cuda/event_pool.hpp>
#else
namespace ghex
{
namespace device
{
struct event_pool
{
  public: // constructors
    event_pool(std::size_t) {}
    event_pool(const event_pool&) = delete;
    event_pool& operator=(const event_pool&) = delete;
    event_pool(event_pool&& other) noexcept = default;
    event_pool& operator=(event_pool&&) noexcept = default;

    void rewind() {}
    void clear() {}
};
} // namespace device
} // namespace ghex
#endif
