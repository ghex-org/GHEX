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
#ifndef INCLUDED_GHEX_UTILS_HPP
#define INCLUDED_GHEX_UTILS_HPP

template<typename Msg>
void make_zero(Msg& msg) {
    for (auto& c : msg)
	c = 0;
}

#endif /* INCLUDED_GHEX_UTILS_HPP */

