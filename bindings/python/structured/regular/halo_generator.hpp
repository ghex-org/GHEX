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

#include <gridtools/common/for_each.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <structured/types.hpp>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{
template<int D>
using halo_generator_args_ =
    gridtools::meta::cartesian_product<types::domain_ids, gridtools::meta::list<types::dims_<D>>>;

template<int D>
using halo_generator_specializations_ = gridtools::meta::transform<
    gridtools::meta::rename<ghex::structured::regular::halo_generator>::template apply,
    halo_generator_args_<D>>;

using halo_generator_specializations = gridtools::meta::concat<halo_generator_specializations_<2>,
    halo_generator_specializations_<3>, halo_generator_specializations_<4>>;
} // namespace
} // namespace regular
} // namespace structured
} // namespace pyghex
