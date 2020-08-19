/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_RMA_ACCESS_MODE_HPP
#define INCLUDED_GHEX_RMA_ACCESS_MODE_HPP

namespace gridtools {
namespace ghex {
namespace rma {

enum class access_mode : int
{
    local,
    remote
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_ACCESS_MODE_HPP */

