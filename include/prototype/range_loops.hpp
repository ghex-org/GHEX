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

#include <array>
#include <tuple>
#include <iomanip>
#include <iostream>

namespace gridtools {

    namespace _impl {

        template <int NestingLevel, typename Ranges, typename Indices, typename Fun>
        typename std::enable_if<NestingLevel == std::tuple_size<Ranges>::value, void>::type
        iterate(Ranges const& ranges, Indices & indices, Fun && fun) {
            fun(indices);
        }

        template <int NestingLevel, typename Ranges, typename Indices, typename Fun>
        typename std::enable_if<NestingLevel != std::tuple_size<Ranges>::value, void>::type
        iterate(Ranges const& ranges, Indices & indices, Fun && fun) {
            auto const& r = std::get<NestingLevel>(ranges);
            for (int i = r.begin(); i < r.end(); ++i) {
                indices[NestingLevel] = i;

                iterate<NestingLevel+1>(ranges, indices, std::forward<Fun>(fun));
            }
        }

        template <int NestingLevel, typename Ranges>
        constexpr typename std::enable_if<NestingLevel == std::tuple_size<Ranges>::value, size_t>::type
        get_size(Ranges const& ranges) {
            return 1;
        }

        template <int NestingLevel, typename Ranges>
        constexpr typename std::enable_if<NestingLevel != std::tuple_size<Ranges>::value, size_t>::type
        get_size(Ranges const& ranges) {
            auto const& r = std::get<NestingLevel>(ranges);
            return (r.end()-r.begin()) * get_size<NestingLevel+1>(ranges);
        }

    } //namespace _impl

    template <typename Ranges, typename Fun>
    void range_loop(Ranges const& ranges, Fun && fun) {
        std::array<int, std::tuple_size<Ranges>::value> indices;

        _impl::iterate<0>(ranges, indices, std::forward<Fun>(fun));
    }

    template <typename Ranges>
    constexpr size_t range_loop_size(Ranges const& ranges) {

        return _impl::get_size<0>(ranges);
    }
} // namespace gridtools
