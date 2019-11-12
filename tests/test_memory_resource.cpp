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

#include <ghex/allocator/memory_resource.hpp>
#include <ghex/transport_layer/message_buffer.hpp>
#include <gtest/gtest.h>
#include <vector>
#include <iostream>

template<typename T>
void test_simple_message(int n)
{
    using namespace gridtools::ghex;

    using allocator_type = allocator::polymorphic_allocator<T>;

    allocator_type alloc( allocator::new_delete_resource() );

    tl::message_buffer<allocator_type> msg(n*sizeof(T), alloc);

}

template<typename T, std::size_t Alignment>
void test_simple_aligned_message(int n)
{
    using namespace gridtools::ghex;

    using allocator_type = allocator::aligned_polymorphic_allocator<T,Alignment>;

    allocator_type alloc( allocator::new_delete_resource() );

    tl::message_buffer<allocator_type> msg(n*sizeof(T), alloc);
    
    EXPECT_TRUE((reinterpret_cast<std::uintptr_t>(msg.data()) & std::uintptr_t(Alignment-1u)) == 0u);
}

TEST(simple_message, integer)
{
    test_simple_message<int>(1);
    test_simple_message<int>(1);
    test_simple_message<int>(1);
    test_simple_message<int>(1);
    test_simple_message<int>(19);
    test_simple_message<int>(19);
    test_simple_message<int>(19);
    test_simple_message<int>(19);
}

TEST(simple_aligned_message, integer)
{
    test_simple_aligned_message<int,16>(1);
    test_simple_aligned_message<int,16>(1);
    test_simple_aligned_message<int,16>(1);
    test_simple_aligned_message<int,16>(1);
    test_simple_aligned_message<int,16>(19);
    test_simple_aligned_message<int,16>(19);
    test_simple_aligned_message<int,16>(19);
    test_simple_aligned_message<int,16>(19);
}

