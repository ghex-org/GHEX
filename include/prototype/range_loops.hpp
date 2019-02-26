/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <array>
#include <tuple>
#include <iomanip>
#include <iostream>

namespace gridtools {

    namespace _impl {

        template <int NesingLevel, typename Ranges, typename Indices, typename Fun>
        typename std::enable_if<NesingLevel == std::tuple_size<Ranges>::value, void>::type
        iterate(Ranges const& ranges, Indices & indices, Fun && fun) {
            fun(indices);
        }

        template <int NesingLevel, typename Ranges, typename Indices, typename Fun>
        typename std::enable_if<NesingLevel != std::tuple_size<Ranges>::value, void>::type
        iterate(Ranges const& ranges, Indices & indices, Fun && fun) {
            auto const& r = std::get<NesingLevel>(ranges);
            for (int i = r.begin(); i < r.end(); ++i) {
                indices[NesingLevel] = i;

                iterate<NesingLevel+1>(ranges, indices, std::forward<Fun>(fun));
            }
        }

    } //namespace _impl

    template <typename Ranges, typename Fun>
    void range_loop(Ranges const& ranges, Fun && fun) {
        std::array<int, std::tuple_size<Ranges>::value> indices;

        _impl::iterate<0>(ranges, indices, std::forward<Fun>(fun));
    }
} // namespace gridtools
