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
#ifndef INCLUDED_GHEX_TL_UCX_ERROR_HPP
#define INCLUDED_GHEX_TL_UCX_ERROR_HPP

#include <string>
#include <stdexcept>
#include <ucp/api/ucp.h>

#ifdef NDEBUG
    #define GHEX_CHECK_UCX_RESULT(x) x;
#else
    #define GHEX_CHECK_UCX_RESULT(x) \
    if (x != UCS_OK)           \
        throw std::runtime_error("GHEX Error: UCX Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif

#endif /* INCLUDED_GHEX_TL_UCX_ERROR_HPP */

