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

namespace ghex
{
namespace util
{
/** @brief compile-time exponentiation of an integer base with positive integer exponent */
inline constexpr int
ct_pow(int base, int exp)
{
    return exp == 0 ? 1 : base * ct_pow(base, exp - 1);
}

} //namespace util
} // namespace ghex
