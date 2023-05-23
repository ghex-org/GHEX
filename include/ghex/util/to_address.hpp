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

#include <type_traits>

namespace ghex
{
namespace util
{
template<class T>
constexpr T*
to_address(T* p) noexcept
{
    static_assert(!std::is_function<T>::value, "T cannot be a function");
    return p;
}

template<class T>
auto
to_address(T& p) noexcept
{
    return to_address(p.operator->());
}

} // namespace util
} // namespace ghex
