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

#include <ghex/allocator/aligned_allocator_adaptor.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

template<std::size_t Alignment, typename T>
void test_std_alloc(int n)
{
    using namespace gridtools::ghex;
    using alloc_t = std::allocator<T>;
    using alloc_other_t = std::allocator<float>;
    using aligned_alloc_t = allocator::aligned_allocator_adaptor<alloc_t, Alignment>;
    using aligned_alloc_other_t = allocator::aligned_allocator_adaptor<alloc_other_t, Alignment>;

    alloc_t alloc;
    aligned_alloc_other_t aligned_alloc_other(alloc);
    aligned_alloc_t aligned_alloc(std::move(aligned_alloc_other));
    
    auto ptr = aligned_alloc.allocate(n);
    EXPECT_TRUE( aligned_alloc_t::offset == ((Alignment<=alignof(std::max_align_t))
                                             ? 0u 
                                             : (alignof(std::max_align_t) + (Alignment-alignof(std::max_align_t)))));
    
    aligned_alloc.construct(ptr, Alignment);
    
    EXPECT_TRUE(ptr[0] == Alignment);
    EXPECT_TRUE((reinterpret_cast<std::uintptr_t>(to_address(ptr)) & std::uintptr_t(Alignment-1u)) == 0u);
    
    for (int i=0; i<n; ++i)
        aligned_alloc.destroy(ptr+i);

    aligned_alloc.deallocate(ptr,n);


    std::vector<T, aligned_alloc_t> vec(n, Alignment, aligned_alloc);

    EXPECT_TRUE(vec[0] == Alignment);
    EXPECT_TRUE((reinterpret_cast<std::uintptr_t>(&vec[0]) & std::uintptr_t(Alignment-1u)) == 0u);
}

TEST(aligned_allocator, integer)
{
    test_std_alloc< 16,int>(1);
    test_std_alloc< 32,int>(1);
    test_std_alloc< 64,int>(1);
    test_std_alloc<128,int>(1);
    test_std_alloc< 16,int>(19);
    test_std_alloc< 32,int>(19);
    test_std_alloc< 64,int>(19);
    test_std_alloc<128,int>(19);
}

TEST(aligned_allocator, double_prec)
{
    test_std_alloc< 16,double>(1);
    test_std_alloc< 32,double>(1);
    test_std_alloc< 64,double>(1);
    test_std_alloc<128,double>(1);
    test_std_alloc< 16,double>(19);
    test_std_alloc< 32,double>(19);
    test_std_alloc< 64,double>(19);
    test_std_alloc<128,double>(19);
}

