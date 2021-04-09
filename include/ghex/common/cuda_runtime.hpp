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

#include <gridtools/common/cuda_runtime.hpp>

/* additional cuda -> hip translations */
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventInterprocess hipEventInterprocess
#define cudaEventQuery hipEventQuery
#define cudaIpcCloseMemHandle hipIpcCloseMemHandle
#define cudaIpcEventHandle_t hipIpcEventHandle_t
#define cudaIpcGetEventHandle hipIpcGetEventHandle
#define cudaIpcGetMemHandle hipIpcGetMemHandle
#define cudaIpcMemHandle_t hipIpcMemHandle_t
#define cudaIpcMemLazyEnablePeerAccess hipIpcMemLazyEnablePeerAccess
#define cudaIpcOpenEventHandle hipIpcOpenEventHandle
#define cudaIpcOpenMemHandle hipIpcOpenMemHandle
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking

#endif /* INCLUDED_GHEX_COMMON_CUDA_RUNTIME_HPP */
