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

#if defined(__NVCC__)
#define GHEX_FORCE_INLINE __forceinline__
#define GHEX_FORCE_INLINE_LAMBDA
#elif defined(__GNUC__)
#define GHEX_FORCE_INLINE        inline __attribute__((always_inline))
#define GHEX_FORCE_INLINE_LAMBDA __attribute__((always_inline))
#elif defined(_MSC_VER)
#define GHEX_FORCE_INLINE inline __forceinline
#define GHEX_FORCE_INLINE_LAMBDA
#else
#define GHEX_FORCE_INLINE inline
#define GHEX_FORCE_INLINE_LAMBDA
#endif

#ifdef GHEX_CUDACC
#define GHEX_HOST_DEVICE __host__ __device__
#ifdef __NVCC__ // NVIDIA CUDA compilation
#define GHEX_DEVICE __device__
#define GHEX_HOST   __host__
#else // Clang CUDA or HIP compilation
#define GHEX_DEVICE __device__ __host__
#define GHEX_HOST   __host__
#endif
#else
#define GHEX_HOST_DEVICE
#define GHEX_HOST
#endif

#ifndef GHEX_FUNCTION
#define GHEX_FUNCTION GHEX_HOST_DEVICE GHEX_FORCE_INLINE
#endif

#ifndef GHEX_FUNCTION_HOST
#define GHEX_FUNCTION_HOST GHEX_HOST GHEX_FORCE_INLINE
#endif

#ifndef GHEX_FUNCTION_DEVICE
#define GHEX_FUNCTION_DEVICE GHEX_DEVICE GHEX_FORCE_INLINE
#endif
