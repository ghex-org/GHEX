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
#ifndef INCLUDED_GHEX_ALLOCATOR_CUDA_ALLOCATOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_CUDA_ALLOCATOR_HPP

#include <memory>
#include "../cuda_utils/error.hpp"

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC
#include "../common/cuda_runtime.hpp"
#endif

#ifdef GHEX_CUDACC
namespace gridtools {
    namespace ghex {
        namespace allocator {

            namespace cuda {

                template<typename T>
                struct allocator
                {
                    using size_type = std::size_t;
                    using value_type = T;
                    using traits = std::allocator_traits<allocator<T>>;
                    using is_always_equal = std::true_type;

                    allocator() noexcept {}
                    template<typename U>
                    allocator(const allocator<U>&) noexcept {}
                    template<typename U>
                    allocator(allocator<U>&&) noexcept {}

                    [[nodiscard]] T* allocate(size_type n, const void* cvptr = nullptr)
                    {
                        T* ptr = nullptr;
                        GHEX_CHECK_CUDA_RESULT(cudaMalloc((void**)&ptr, n*sizeof(T)));
                        return ptr;
                    }

                    // WARN: temporarily leaking memory due to MPICH bug
                    void deallocate(T* ptr, size_type n)
                    {
                        // not freeing because of CRAY-BUG
                        // GHEX_CHECK_CUDA_RESULT(cudaFree(ptr));
                    }

                    void swap(const allocator&) {}

                    friend bool operator==(const allocator&, const allocator&) { return true; }
                    friend bool operator!=(const allocator&, const allocator&) { return false; }
                };

            } // namespace cuda

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools
#endif

#endif /* INCLUDED_GHEX_ALLOCATOR_CUDA_ALLOCATOR_HPP */

