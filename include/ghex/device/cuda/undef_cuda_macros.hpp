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

/* GridTools cuda -> hip translations */
#undef cudaDeviceProp
#undef cudaDeviceSynchronize
#undef cudaErrorInvalidValue
#undef cudaError_t
#undef cudaEventCreate
#undef cudaEventDestroy
#undef cudaEventElapsedTime
#undef cudaEventRecord
#undef cudaEventSynchronize
#undef cudaEvent_t
#undef cudaFree
#undef cudaFreeHost
#undef cudaGetDevice
#undef cudaGetDeviceCount
#undef cudaGetDeviceProperties
#undef cudaGetErrorName
#undef cudaGetErrorString
#undef cudaGetLastError
#undef cudaMalloc
#undef cudaMallocHost
#undef cudaMallocManaged
#undef cudaMemAttachGlobal
#undef cudaMemcpy
#undef cudaMemcpyDeviceToHost
#undef cudaMemcpyHostToDevice
#undef cudaMemoryTypeDevice
#undef cudaPointerAttributes
#undef cudaPointerGetAttributes
#undef cudaSetDevice
#undef cudaStreamCreate
#undef cudaStreamDestroy
#undef cudaStreamSynchronize
#undef cudaStream_t
#undef cudaSuccess

/* additional cuda -> hip translations */
#undef cudaEventCreateWithFlags
#undef cudaEventDisableTiming
#undef cudaEventInterprocess
#undef cudaEventQuery
#undef cudaIpcCloseMemHandle
#undef cudaIpcEventHandle_t
#undef cudaIpcGetEventHandle
#undef cudaIpcGetMemHandle
#undef cudaIpcMemHandle_t
#undef cudaIpcMemLazyEnablePeerAccess
#undef cudaIpcOpenEventHandle
#undef cudaIpcOpenMemHandle
#undef cudaMemcpyAsync
#undef cudaStreamCreateWithFlags
#undef cudaStreamNonBlocking
