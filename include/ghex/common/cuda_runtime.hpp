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
#ifndef INCLUDED_GHEX_COMMON_CUDA_RUNTIME_HPP
#define INCLUDED_GHEX_COMMON_CUDA_RUNTIME_HPP

// if necessary, this define might be moved to a defs.hpp,
// or the GridTools one might be used instead
#if (defined(__CUDACC__) || defined(__HIP__))
#define GHEX_CUDACC
#endif

#include <gridtools/common/cuda_runtime.hpp>

// TO DO: add missing cuda -> hip translations when necessary

#endif /* INCLUDED_GHEX_COMMON_CUDA_RUNTIME_HPP */
