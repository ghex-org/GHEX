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

#include <vector>
#include <structured/types.hpp>
#include <structured/regular/domain_descriptor.hpp>
#include <structured/regular/halo_generator.hpp>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{

template<typename DomainId, typename Dimension>
struct make_pattern_traits
{
    using halo_gen = ghex::structured::regular::halo_generator<DomainId, Dimension>;
    using domain_desc = ghex::structured::regular::domain_descriptor<DomainId, Dimension>;
    using domain_range = std::vector<domain_desc>;
};

template<int D>
using make_pattern_traits_args_ =
    gridtools::meta::cartesian_product<types::domain_ids, gridtools::meta::list<types::dims_<D>>>;

template<int D>
using make_pattern_traits_specializations_ =
    gridtools::meta::transform<gridtools::meta::rename<make_pattern_traits>::template apply,
        make_pattern_traits_args_<D>>;

using make_pattern_traits_specializations =
    gridtools::meta::concat<make_pattern_traits_specializations_<2>,
        make_pattern_traits_specializations_<3>, make_pattern_traits_specializations_<4>>;

} // namespace
} // namespace regular
} // namespace structured
} // namespace pyghex
