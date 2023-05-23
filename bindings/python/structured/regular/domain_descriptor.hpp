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

#include <gridtools/meta.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
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
using domain_descriptor_args_ =
    gridtools::meta::cartesian_product<types::domain_ids, gridtools::meta::list<types::dims_<D>>>;

template<int D>
using domain_descriptor_specializations_ = gridtools::meta::transform<
    gridtools::meta::rename<ghex::structured::regular::domain_descriptor>::template apply,
    domain_descriptor_args_<D>>;
} // namespace

using domain_descriptor_specializations =
    gridtools::meta::concat<domain_descriptor_specializations_<2>,
        domain_descriptor_specializations_<3>, domain_descriptor_specializations_<4>>;

} // namespace regular
} // namespace structured
} // namespace pyghex
