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
#ifndef INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP

#include <cstdlib>
#include <new>
#include <stdexcept>
#include <limits>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace gridtools {
    namespace ghex {
        namespace allocator {

            template <class T>
            struct unified_memory_allocator {

                typedef T value_type;

                unified_memory_allocator() = default;

                template <class U>
                constexpr unified_memory_allocator(const unified_memory_allocator<U>&) noexcept {}

                [[nodiscard]] T* allocate(std::size_t n) {
                    if (n > std::numeric_limits<std::size_t>::max() / sizeof(T)) throw std::bad_alloc();
                    T* p;
#ifdef __CUDACC__
                    cudaError_t err = cudaMallocManaged(reinterpret_cast<void**>(&p), n * sizeof(T));
                    if (err != cudaSuccess) throw std::runtime_error("ERROR: failed to allocate GPU memory");
#else
                    if (!(p = reinterpret_cast<T*>(std::malloc(n * sizeof(T))))) throw std::bad_alloc();
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
                    std::free(reinterpret_cast<void*>(p));
#endif
                }

            };

            template <class T, class U>
            bool operator==(const unified_memory_allocator<T>&, const unified_memory_allocator<U>&) { return true; }

            template <class T, class U>
            bool operator!=(const unified_memory_allocator<T>&, const unified_memory_allocator<U>&) { return false; }

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_UNIFIED_MEMORY_ALLOCATOR_HPP */
