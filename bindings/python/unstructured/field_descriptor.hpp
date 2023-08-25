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
#include <ghex/unstructured/user_concepts.hpp>
#include <unstructured/types.hpp>
#include <unstructured/domain_descriptor.hpp>

namespace pyghex
{
namespace unstructured
{
namespace
{
using field_descriptor_args = gridtools::meta::cartesian_product<types::archs,
types::domain_ids, types::global_ids, types::data>;

using field_descriptor_specializations = gridtools::meta::transform<
    gridtools::meta::rename<ghex::unstructured::data_descriptor>::template apply,
    field_descriptor_args>;
} // namespace
} // namespace unstructured
} // namespace pyghex

