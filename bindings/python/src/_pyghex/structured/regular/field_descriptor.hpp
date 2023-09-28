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

#include <ghex/structured/regular/field_descriptor.hpp>
#include <structured/types.hpp>
#include <structured/regular/domain_descriptor.hpp>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{
template<int D>
using field_descriptor_args_ = gridtools::meta::cartesian_product<types::data, types::archs,
    domain_descriptor_specializations_<D>, types::layouts_<D>>;

template<int D>
using field_descriptor_specializations_ = gridtools::meta::transform<
    gridtools::meta::rename<ghex::structured::regular::field_descriptor>::template apply,
    field_descriptor_args_<D>>;

using field_descriptor_specializations =
    gridtools::meta::concat<field_descriptor_specializations_<2>,
        field_descriptor_specializations_<3>, field_descriptor_specializations_<4>>;

} // namespace
} // namespace regular
} // namespace structured
} // namespace pyghex
