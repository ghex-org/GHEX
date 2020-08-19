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
#ifndef INCLUDED_GHEX_RMA_LOCALITY_HPP
#define INCLUDED_GHEX_RMA_LOCALITY_HPP

#include "./thread/access_guard.hpp"
#ifdef GHEX_USE_XPMEM
#include "./xpmem/access_guard.hpp"
#endif

namespace gridtools {
namespace ghex {
namespace rma {

enum class locality
{
    thread,
    process,
    remote
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_LOCALITY_HPP */
