/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <tuple>


namespace ghex {

    template<std::size_t I = 0, typename Func, typename ...Args>
    typename std::enable_if<I == sizeof...(Args)>::type for_each(std::tuple<Args...>& t, Func) {}

    template<std::size_t I = 0, typename Func, typename ...Args>
    typename std::enable_if<(I < sizeof...(Args))>::type for_each(std::tuple<Args...>& t, Func f) {
        f(std::get<I>(t));
        for_each<I+1, Func, Args...>(t, f);
    }

}
