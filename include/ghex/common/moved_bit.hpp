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
#ifndef INCLUDED_GHEX_COMMON_MOVED_BIT_HPP
#define INCLUDED_GHEX_COMMON_MOVED_BIT_HPP

namespace gridtools {
    namespace ghex {

        struct moved_bit
        {
            bool m_moved = false;

            moved_bit() = default;
            moved_bit(bool state) noexcept : m_moved{state} {}
            moved_bit(const moved_bit &) = default;
            moved_bit(moved_bit &&other) noexcept
            : m_moved{std::exchange(other.m_moved, true)} 
            {}

            moved_bit &operator=(const moved_bit &) = default;
            moved_bit &operator=(moved_bit &&other) noexcept
            {
                m_moved = std::exchange(other.m_moved, true);
                return *this;
            }

            operator bool() const { return m_moved; }
        };

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMON_MOVED_BIT_HPP */

