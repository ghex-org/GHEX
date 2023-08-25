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

namespace ghex
{
namespace unstructured
{
namespace detail
{
template<typename Index>
struct grid
{
    using index_type = Index;
};

} // namespace detail

/** @brief type to indicate unstructured grids */
struct grid
{
    template<typename Domain>
    using type = detail::grid<typename Domain::local_index_type>;
};

} // namespace unstructured
} // namespace ghex
