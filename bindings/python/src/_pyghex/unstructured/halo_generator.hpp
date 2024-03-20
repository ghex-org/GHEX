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

#include <ghex/unstructured/user_concepts.hpp>
#include <unstructured/types.hpp>

namespace pyghex
{
namespace unstructured
{
namespace
{
using halo_generator_args =
    gridtools::meta::cartesian_product<types::domain_ids, types::global_ids>;

using halo_generator_specializations = gridtools::meta::transform<
    gridtools::meta::rename<ghex::unstructured::halo_generator>::template apply,
    halo_generator_args>;
} // namespace
} // namespace unstructured
} // namespace pyghex

