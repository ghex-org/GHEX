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
#include <cstdint>
#include <gridtools/meta.hpp>
#include <ghex/config.hpp>

namespace pyghex
{

struct types
{
    using archs = gridtools::meta::rename<gridtools::meta::list, ghex::arch_list>;
    using data = gridtools::meta::list<double, float, bool, std::int32_t, std::int64_t>;
    using domain_ids = gridtools::meta::list<int>;
};

} // namespace pyghex
