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

#include <utility>

namespace ghex
{
namespace util
{
struct moved_bit
{
    bool m_moved = false;

    moved_bit() = default;
    moved_bit(bool state) noexcept
    : m_moved{state}
    {
    }
    moved_bit(const moved_bit&) = default;
    moved_bit(moved_bit&& other) noexcept
    : m_moved{std::exchange(other.m_moved, true)}
    {
    }

    moved_bit& operator=(const moved_bit&) = default;
    moved_bit& operator=(moved_bit&& other) noexcept
    {
        m_moved = std::exchange(other.m_moved, true);
        return *this;
    }

    operator bool() const { return m_moved; }
};

} // namespace util
} // namespace ghex
