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
#include <ghex/unstructured/pattern.hpp>
#include <unstructured/types.hpp>
#include <unstructured/domain_descriptor.hpp>
#include <unstructured/halo_generator.hpp>

namespace pyghex
{
namespace unstructured
{
namespace
{
template<typename Index, typename DomainId>
struct make_pattern_traits
{
    using halo_gen = ghex::unstructured::halo_generator<DomainId, Index>;
    using domain_desc = ghex::unstructured::domain_descriptor<DomainId, Index>;
    using domain_range = std::vector<domain_desc>;
};

using make_pattern_traits_args =
    gridtools::meta::cartesian_product<types::domain_ids, types::global_ids>;

using make_pattern_traits_specializations =
    gridtools::meta::transform<gridtools::meta::rename<make_pattern_traits>::template apply,
        make_pattern_traits_args>;
} // namespace
} // namespace unstructured
} // namespace pyghex


