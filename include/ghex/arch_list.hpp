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
#ifndef INCLUDED_GHEX_ARCH_LIST_HPP
#define INCLUDED_GHEX_ARCH_LIST_HPP

#include <tuple>

namespace gridtools {
    namespace ghex {

        struct cpu {};
        struct gpu {};

#ifdef __CUDACC__
            using arch_list = std::tuple<cpu,gpu>;
#else
#ifdef GHEX_EMULATE_GPU
            using arch_list = std::tuple<cpu,gpu>;
#else
            using arch_list = std::tuple<cpu>;
#endif
#endif

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ARCH_LIST_HPP */

