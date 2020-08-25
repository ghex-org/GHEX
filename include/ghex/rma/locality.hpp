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

namespace gridtools {
namespace ghex {
namespace rma {

// source/destination of a communication
// can be either
// - among threads of the same rank,
// - among ranks on the same shared memory region
// - or remote
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
