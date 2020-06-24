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
#ifndef INCLUDED_GHEX_REMOTE_RANGE_TRAITS_HPP
#define INCLUDED_GHEX_REMOTE_RANGE_TRAITS_HPP

namespace gridtools {
namespace ghex {

template<template<typename> typename RangeGen>
struct remote_range_traits
{
    template<typename Communicator>
    static bool is_local(Communicator comm, int remote_rank) { return false; }
};

} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_REMOTE_RANGE_TRAITS_HPP */
