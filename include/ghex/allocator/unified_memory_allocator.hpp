/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP

#include <stdexcept>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gridtools {
    namespace ghex {
        namespace allocator {

            // WARN: some member types and member functions provided in std::allocator<T> are still not provided here;
            // WARN: fails to compile is inheritance is used (potential issues on vector copy constructor when using different allocators)
            template <class T>
            struct unified_memory_allocator {

                typedef T value_type;

                [[nodiscard]] T* allocate(std::size_t n) {
                    T* p;
#ifdef __CUDACC__
                    cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&p), n * sizeof(T));
                    if (err != cudaSuccess) throw std::runtime_error("ERROR: failed to allocate GPU memory");
#else
                    p = reinterpret_cast<T*>(malloc(n * sizeof(T)));
#endif
                    return p;
                }

                void deallocate(T* p, std::size_t) {
#ifdef __CUDACC__
                    cudaError_t err;
                    err = cudaDeviceSynchronize();
                    if (err != cudaSuccess) throw std::runtime_error("ERROR: failed to synchronize GPU memory");
                    err = cudaFree(reinterpret_cast<void*>(p));
                    if (err != cudaSuccess) throw std::runtime_error("ERROR: failed to free GPU memory");
#else
                    free(reinterpret_cast<void*>(p));
#endif
                }

            };

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP */
