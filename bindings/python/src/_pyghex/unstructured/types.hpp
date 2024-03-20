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
#include <ghex/unstructured/grid.hpp>
#include <types.hpp>

namespace pyghex
{

namespace unstructured
{

struct types : public ::pyghex::types
{
    using global_ids = gridtools::meta::list<int>;
    using grids = gridtools::meta::list<ghex::unstructured::detail::grid<std::size_t> >;
};

} // namespace unstructured

} // namespace pyghex
