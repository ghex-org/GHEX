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

#define GHEX_VERSION 600
#define GHEX_VERSION_MAJOR 0
#define GHEX_VERSION_MINOR 6
#define GHEX_VERSION_PATCH 0

#include <oomph/config.hpp>
#include <tuple>

/* #undef GHEX_NO_RMA */
/* #undef GHEX_USE_XPMEM */
/* #undef GHEX_USE_XPMEM_ACCESS_GUARD */
/* #undef GHEX_USE_GPU */
#define GHEX_GPU_MODE none
/* #undef GHEX_GPU_MODE_EMULATE */
#define GHEX_DEVICE_NONE
/* #undef GHEX_COMM_OBJ_USE_U */
/* #undef GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST */
/* #undef GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST */

// detect hip-clang and set macros (see hip/hip_common.h)
#if defined(GHEX_DEVICE_HIP)
#   if defined(__clang__) && defined(__HIP__)
#       ifndef __HIP_PLATFORM_AMD__
#           define __HIP_PLATFORM_AMD__
#       endif
#   endif
#   if defined(__NVCC__) || (defined(__clang__) && defined(__CUDA__) && !defined(__HIP__))
#       ifndef __HIP_PLATFORM_NVIDIA__
#           define __HIP_PLATFORM_NVIDIA__
#       endif
#       ifdef __CUDACC__
#           define __HIPCC__
#       endif
#   endif
#endif

#if defined(GHEX_USE_GPU)
#   if (defined(__CUDACC__) || defined(__HIP_PLATFORM_AMD__))
#       define GHEX_CUDACC
#   endif
#endif

namespace ghex
{
struct cpu
{
};
struct gpu
{
};

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
using arch_list = std::tuple<cpu, gpu>;
#else
using arch_list = std::tuple<cpu>;
#endif

#if defined(GHEX_DEVICE_NONE) && !defined(OOPMPH_DEVICE_NONE)
#   pragma error "oomph (dependency) differs in GPU support"
#elif defined(GHEX_DEVICE_CUDA) && !defined(OOMPH_DEVICE_CUDA)
#   pragma error "oomph (dependency) differs in GPU support"
#elif defined(GHEX_DEVICE_HIP) && !defined(OOMPH_DEVICE_HIP)
#   pragma error "oomph (dependency) differs in GPU support"
#elif defined(GHEX_DEVICE_EMULATE) && !defined(OOMPH_DEVICE_EMULATE)
#   pragma error "oomph (dependency) differs in GPU support"
#endif

} // namespace ghex
