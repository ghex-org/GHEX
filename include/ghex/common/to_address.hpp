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
#ifndef INCLUDED_GHEX_COMMON_TO_ADDRESS_HPP
#define INCLUDED_GHEX_COMMON_TO_ADDRESS_HPP

#include <type_traits>

namespace gridtools {
    namespace ghex {

        template<class T>
        constexpr T* to_address(T* p) noexcept
        {
            static_assert(!std::is_function<T>::value, "T cannot be a function");
            return p;
        }

        template<class T>
        auto to_address(T& p) noexcept
        {
            return to_address(p.operator->());
        }

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MESSAGE_BUFFER_HPP */

