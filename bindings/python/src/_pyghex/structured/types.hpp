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
#include <array>
#include <gridtools/common/layout_map.hpp>
#include <ghex/structured/grid.hpp>
#include <types.hpp>

namespace pyghex
{

namespace structured
{

namespace detail
{

template<int D>
using dims = std::integral_constant<int, D>;

template<int>
struct layouts;

template<>
struct layouts<2>
{
    using list = gridtools::meta::list<gridtools::layout_map<0, 1>, gridtools::layout_map<1, 0>>;
};

template<>
struct layouts<3>
{
    using list =
        gridtools::meta::list<gridtools::layout_map<0, 1, 2>, gridtools::layout_map<0, 2, 1>,
            gridtools::layout_map<1, 0, 2>, gridtools::layout_map<1, 2, 0>,
            gridtools::layout_map<2, 0, 1>, gridtools::layout_map<2, 1, 0>>;
};

template<>
struct layouts<4>
{
    using list =
        gridtools::meta::list<gridtools::layout_map<0, 1, 2, 3>, gridtools::layout_map<0, 1, 3, 2>,
            gridtools::layout_map<0, 2, 1, 3>, gridtools::layout_map<0, 2, 3, 1>,
            gridtools::layout_map<0, 3, 1, 2>, gridtools::layout_map<0, 3, 2, 1>,
            gridtools::layout_map<1, 0, 2, 3>, gridtools::layout_map<1, 0, 3, 2>,
            gridtools::layout_map<1, 2, 0, 3>, gridtools::layout_map<1, 2, 3, 0>,
            gridtools::layout_map<1, 3, 0, 2>, gridtools::layout_map<1, 3, 2, 0>,
            gridtools::layout_map<3, 0, 1, 2>, gridtools::layout_map<3, 0, 2, 1>,
            gridtools::layout_map<3, 1, 0, 2>, gridtools::layout_map<3, 1, 2, 0>,
            gridtools::layout_map<3, 2, 0, 1>, gridtools::layout_map<3, 2, 1, 0>>;
};

template<int D>
using grids = ghex::structured::detail::grid<std::array<int, D>>;

} // namespace detail

struct types : public ::pyghex::types
{
    template<int D>
    using dims_ = detail::dims<D>;
    using dims = gridtools::meta::list<dims_<2>, dims_<3>, dims_<4>>;

    template<int D>
    using grids_ = detail::grids<D>;
    using grids = gridtools::meta::list<grids_<2>, grids_<3>, grids_<4>>;

    template<int D>
    using layouts_ = typename detail::layouts<D>::list;
    using layouts = gridtools::meta::list<layouts_<2>, layouts_<3>, layouts_<4>>;
};

} // namespace structured

} // namespace pyghex
