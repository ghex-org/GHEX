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
#include <ghex/communication_object.hpp>
#include <unstructured/types.hpp>

namespace pyghex
{
namespace unstructured
{
namespace
{
using communication_object_args =
    gridtools::meta::cartesian_product<types::grids, types::domain_ids>;

using communication_object_specializations =
    gridtools::meta::transform<gridtools::meta::rename<ghex::communication_object>::template apply,
        communication_object_args>;
} // namespace
} // namespace unstructured
} // namespace pyghex

