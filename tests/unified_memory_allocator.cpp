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
#include <vector>
#include <gtest/gtest.h>
#include <ghex/allocator/unified_memory_allocator.hpp>

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <ghex/common/cuda_runtime.hpp>
#endif


#ifdef GHEX_CUDACC
template <typename T>
__global__ void add_one(T* values, const std::size_t n) {
    auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < n) values[idx] += 1;
}
#endif


template <typename T> // Integer values assumed, type T used basically to test different alignments
void run_test(const std::size_t n) {

    using allocator_type = gridtools::ghex::allocator::cuda::unified_memory_allocator<T>;

    // Allocate on the CPU
    allocator_type alloc{};
    auto ptr = alloc.allocate(n);
    std::vector<T, allocator_type> vec(n);

    // Initialize on the CPU
    for (std::size_t i = 0; i < n; ++i) {
        auto value = static_cast<T>(i%10); // just to avoid casting issues
        ptr[i] = value;
        vec[i] = value;
    }

    // Access from the CPU
    for (std::size_t i = 0; i < n; ++i) {
        auto value = static_cast<T>(i%10); // just to avoid casting issues
        EXPECT_TRUE(ptr[i] == value);
        EXPECT_TRUE(vec[i] == value);
    }

#ifdef GHEX_CUDACC
    // Launch kernels to access from GPU
    add_one<T><<<n/32, 32>>>(ptr, n); // n = k * 32, k = 1, 2, ...
    add_one<T><<<n/32, 32>>>(&(vec[0]), n); // n = k * 32, k = 1, 2, ...
    cudaDeviceSynchronize();

    // Access from the CPU and verify
    for (std::size_t i = 0; i < n; ++i) {
        auto value = static_cast<T>(i%10);
        EXPECT_TRUE(ptr[i] == (value + 1));
        EXPECT_TRUE(vec[i] == (value + 1));
    }
#endif

    // Deallocate
    EXPECT_NO_THROW(alloc.deallocate(ptr, n));

}


TEST(unified_memory_allocator, integer_allocation) {

    run_test<char>(32);
    run_test<char>(4096);
    run_test<short>(32);
    run_test<short>(4096);
    run_test<int>(32);
    run_test<int>(4096);
    run_test<std::size_t>(32);
    run_test<std::size_t>(4096);

}
