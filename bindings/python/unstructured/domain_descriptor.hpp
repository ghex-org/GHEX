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
using domain_descriptor_args =
    gridtools::meta::cartesian_product<types::domain_ids, types::global_ids>;

using domain_descriptor_specializations = gridtools::meta::transform<
    gridtools::meta::rename<ghex::unstructured::domain_descriptor>::template apply,
    domain_descriptor_args>;
} // namespace
} // namespace unstructured
} // namespace pyghex
