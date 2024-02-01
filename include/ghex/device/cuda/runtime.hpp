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

#ifdef __HIP_PLATFORM_AMD__

#include <hip/hip_runtime.h>
#include <ghex/device/cuda/undef_cuda_macros.hpp>

/* GridTools cuda -> hip translations */
#define cudaDeviceProp hipDeviceProp_t
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaError_t hipError_t
#define cudaEventCreate hipEventCreate
#define cudaEventDestroy hipEventDestroy
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEvent_t hipEvent_t
#define cudaFree hipFree
#define cudaFreeHost hipFreeHost
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaGetErrorName hipGetErrorName
#define cudaGetErrorString hipGetErrorString
#define cudaGetLastError hipGetLastError
#define cudaMalloc hipMalloc
#define cudaMallocHost hipMallocHost
#define cudaMallocManaged hipMallocManaged
#define cudaMemAttachGlobal hipMemAttachGlobal
#define cudaMemcpy hipMemcpy
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemoryTypeDevice hipMemoryTypeDevice
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes
#define cudaSetDevice hipSetDevice
#define cudaStreamCreate hipStreamCreate
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaStream_t hipStream_t
#define cudaSuccess hipSuccess

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

#else /* __HIP_PLATFORM_AMD__ */

#include <cuda_runtime.h>

#endif /* __HIP_PLATFORM_AMD__ */
