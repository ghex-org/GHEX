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

#include <ghex/config.hpp>
#include <exception>
#include <stdexcept>
#include <string>
#include <iostream>

#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>

#define GHEX_CHECK_CUDA_RESULT(x)                                                                  \
    if (x != cudaSuccess)                                                                          \
        throw std::runtime_error("GHEX Error: CUDA Call failed " + std::string(#x) + " (" +        \
                                 std::string(cudaGetErrorString(x)) + ") in " +                    \
                                 std::string(__FILE__) + ":" + std::to_string(__LINE__));

#define GHEX_CHECK_CUDA_RESULT_NO_THROW(x)                                                         \
    try { GHEX_CHECK_CUDA_RESULT(x) }                                                              \
    catch (const std::exception& e) {                                                              \
        std::cerr << e.what() << std::endl;                                                        \
        std::terminate();                                                                          \
    }

#endif
