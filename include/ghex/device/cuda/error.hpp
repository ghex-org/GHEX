/* 
 * GridTools
 * 
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#pragma once

#include <ghex/config.hpp>
#include <stdexcept>
#include <string>

#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>

#ifdef NDEBUG
#define GHEX_CHECK_CUDA_RESULT(x) x;
#else
#define GHEX_CHECK_CUDA_RESULT(x)                                                                  \
    if (x != cudaSuccess)                                                                          \
        throw std::runtime_error("GHEX Error: CUDA Call failed " + std::string(#x) + " (" +        \
                                 std::string(cudaGetErrorString(x)) + ") in " +                    \
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif

#endif
