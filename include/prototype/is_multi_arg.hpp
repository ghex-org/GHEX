/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once

#include <utility>
#include <type_traits>

namespace gridtools {

    template <typename T>
    struct is_true : std::true_type {};

    template <typename F, typename Arg>
    using one_arg_ = is_true<decltype(std::declval<F>()(std::declval<Arg>()))>;

    template <typename F>
    using zero_arg_ = is_true<decltype(std::declval<F>()())>;

    template <typename F, typename Arg, typename VOID = void>
    struct one_arg : std::false_type {};

    template <typename F, typename Arg>
    struct one_arg<F, Arg, typename std::enable_if<one_arg_<F, Arg>::value, void>::type> : std::true_type {};

    template <typename F, typename VOID = void>
    struct zero_arg : std::false_type {};

    template <typename F>
    struct zero_arg<F, typename std::enable_if<zero_arg_<F>::value, void>::type> : std::true_type {};



    // Assuming it is multi arg if it is not a null arg or one arg
    template <typename F, typename Arg, typename Void=void>
    struct is_multi_arg : std::true_type {};

    template <typename F, typename Arg>
    struct is_multi_arg<F, Arg, typename std::enable_if<one_arg<F,Arg>::value or zero_arg<F>::value, void >::type > : std::false_type {};
}
