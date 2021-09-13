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
#ifndef INCLUDED_GHEX_ARCH_TRAITS_HPP
#define INCLUDED_GHEX_ARCH_TRAITS_HPP

#include "./allocator/pool_allocator_adaptor.hpp"
#include "./allocator/aligned_allocator_adaptor.hpp"
#include "./allocator/cuda_allocator.hpp"
#include "./transport_layer/message_buffer.hpp"
#include "./arch_list.hpp"

#include "./common/defs.hpp"
#ifdef GHEX_CUDACC
#include "./common/cuda_runtime.hpp"
#endif

namespace gridtools {
    namespace ghex {

        template<typename Arch>
        struct arch_traits;

        template<>
        struct arch_traits<cpu>
        {
            static constexpr const char* name = "CPU";

            using device_id_type          = int;
            using basic_allocator_type    = std::allocator<unsigned char>;
            using pool_type               = allocator::pool<basic_allocator_type>;
            using pool_allocator_type     = typename pool_type::allocator_type;
            
            //using message_allocator_type  = allocator::aligned_allocator_adaptor<std::allocator<unsigned char>,64>;
            using message_allocator_type  = allocator::aligned_allocator_adaptor<pool_allocator_type,64>;
            using message_type            = tl::message_buffer<message_allocator_type>;

            static device_id_type default_id() { return 0; }

            static message_type make_message(pool_type& pool, device_id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),device_id_type>::value, "trick to prevent warnings");
                //return {};
                return { message_allocator_type{pool.get_allocator()} };
            }
        };

#ifdef GHEX_CUDACC
        template<>
        struct arch_traits<gpu>
        {
            static constexpr const char* name = "GPU";

            using device_id_type          = int;
            using basic_allocator_type    = allocator::cuda::allocator<unsigned char>;
            using pool_type               = allocator::pool<basic_allocator_type>;
            using pool_allocator_type     = typename pool_type::allocator_type;

            //using message_allocator_type  = allocator::cuda::allocator<unsigned char>;
            using message_allocator_type  = pool_allocator_type;
            using message_type            = tl::message_buffer<message_allocator_type>;

            static device_id_type default_id() { return 0; }

            static message_type make_message(pool_type& pool, device_id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),device_id_type>::value, "trick to prevent warnings");
                //return {};
                return { message_allocator_type{pool.get_allocator()} };
            }
        };
#else
#ifdef GHEX_EMULATE_GPU
        template<>
        struct arch_traits<gpu> : public arch_traits<cpu> {};
#endif
#endif

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ARCH_TRAITS_HPP */

