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
#ifndef INCLUDED_GHEX_CUDA_ERROR_HPP
#define INCLUDED_GHEX_CUDA_ERROR_HPP

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC

#include <stdexcept>
#include <string>
#include "../common/cuda_runtime.hpp"

#ifdef NDEBUG
    #define GHEX_CHECK_CUDA_RESULT(x) x;
#else
    #define GHEX_CHECK_CUDA_RESULT(x)                                                       \
    if (x != cudaSuccess)                                                                   \
        throw std::runtime_error("GHEX Error: CUDA Call failed " + std::string(#x) + " ("   \
                                 + std::string(cudaGetErrorString( x ))+ ") in "            \
                                 + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif

#endif

#endif // INCLUDED_GHEX_CUDA_ERROR_HPP

