/* 
 * GridTools
 * 
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/context.hpp>

//#include "./allocator/pool_allocator_adaptor.hpp"
//#include "./allocator/aligned_allocator_adaptor.hpp"
//#include "./allocator/cuda_allocator.hpp"
//#include "./transport_layer/message_buffer.hpp"
//#include "./arch_list.hpp"

namespace ghex
{
template<typename Arch>
struct arch_traits;

template<>
struct arch_traits<cpu>
{
    static constexpr const char* name = "CPU";

    using device_id_type = int;
    //using basic_allocator_type = std::allocator<unsigned char>;
    //using pool_type = allocator::pool<basic_allocator_type>;
    //using pool_allocator_type = typename pool_type::allocator_type;

    ////using message_allocator_type  = allocator::aligned_allocator_adaptor<std::allocator<unsigned char>,64>;
    //using message_allocator_type = allocator::aligned_allocator_adaptor<pool_allocator_type, 64>;
    //using message_type = tl::message_buffer<message_allocator_type>;
    using message_type = context::message_type;

    static device_id_type default_id() { return 0; }

    static message_type make_message(
        oomph::communicator& c, std::size_t size, device_id_type index = default_id())
    {
        static_assert(
            std::is_same<decltype(index), device_id_type>::value, "trick to prevent warnings");
        return c.make_buffer<unsigned char>(size);
    }
};

#if HWMALLOC_ENABLE_DEVICE
template<>
struct arch_traits<gpu>
{
    static constexpr const char* name = "GPU";

    using device_id_type = int;
    //using basic_allocator_type = allocator::cuda::allocator<unsigned char>;
    //using pool_type = allocator::pool<basic_allocator_type>;
    //using pool_allocator_type = typename pool_type::allocator_type;

    ////using message_allocator_type  = allocator::cuda::allocator<unsigned char>;
    //using message_allocator_type = pool_allocator_type;
    //using message_type = tl::message_buffer<message_allocator_type>;
    using message_type = context::message_type;

    static device_id_type default_id() { return 0; }

    static message_type make_message(
        oomph::communicator& c, std::size_t size, device_id_type index = default_id())
    {
        static_assert(
            std::is_same<decltype(index), device_id_type>::value, "trick to prevent warnings");
        return c.make_device_buffer<unsigned char>(size, index);
    }
};
#endif

} // namespace ghex
