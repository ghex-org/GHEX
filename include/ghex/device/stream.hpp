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
#include <ghex/device/cuda/stream.hpp>
#else
namespace ghex
{
namespace device
{
struct stream
{
    // default construct
    stream() {}
    stream(bool) {}

    // non-copyable
    stream(const stream&) noexcept = delete;
    stream& operator=(const stream&) = delete;

    // movable
    stream(stream&& other) noexcept = default;
    stream& operator=(stream&&) noexcept = default;

    void sync() {}
};
} // namespace device
} // namespace ghex
#endif
