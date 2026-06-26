/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <hwmalloc/config.hpp>
#include <oomph/types.hpp>

#define OOMPH_ENABLE_DEVICE HWMALLOC_ENABLE_DEVICE
#define OOMPH_DEVICE_RUNTIME HWMALLOC_DEVICE_RUNTIME
#if defined(HWMALLOC_DEVICE_HIP)
#   define OOMPH_DEVICE_HIP
#elif defined(HWMALLOC_DEVICE_CUDA)
#   define OOMPH_DEVICE_CUDA
#elif defined(HWMALLOC_DEVICE_EMULATE)
#   define OOMPH_DEVICE_EMULATE
#else
#   define OOMPH_DEVICE_NONE
#endif

#define OOMPH_USE_FAST_PIMPL 0
#define OOMPH_ENABLE_BARRIER 1
#define OOMPH_RECURSION_DEPTH 20

#define OOMPH_VERSION 500
#define OOMPH_VERSION_MAJOR 0
#define OOMPH_VERSION_MINOR 5
#define OOMPH_VERSION_PATCH 0
