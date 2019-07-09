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
#ifndef INCLUDED_DEVICES_HPP
#define INCLUDED_DEVICES_HPP

//#include "./protocol/mpi_allocator.hpp"
#include <tuple>
#include <boost/align/aligned_allocator_adaptor.hpp>

namespace gridtools {

    namespace device {

        // Allocator adaptor that interposes construct() calls to
        // convert value initialization into default initialization.
        template <typename T, typename A=std::allocator<T>>
        class default_init_allocator : public A 
        {
            typedef std::allocator_traits<A> a_t;

        public:
            template <typename U> struct rebind {
                using other = default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
            };

            using A::A;

            template <typename U>
            void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) 
            {
                ::new(static_cast<void*>(ptr)) U;
            }
          
            template <typename U, typename...Args>
            void construct(U* ptr, Args&&... args) 
            {
                a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
            }
        };

        struct cpu
        {
            using id_type = int;

            static constexpr const char* name = "CPU";

            static id_type default_id() { return 0; }

            template<typename T>
            //using allocator_type = std::allocator<T>;
            using allocator_type = default_init_allocator<T>;
            //using allocator_type = protocol::mpi_allocator<T>;

            template<typename T>
            using aligned_allocator_type = boost::alignment::aligned_allocator_adaptor<allocator_type<T>, 64>;
            
            template<typename T>
            using vector_type = std::vector<T, aligned_allocator_type<T>>;

            /*struct handle
            {
                void wait() {}
            };*/

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                return vector_type<T>{aligned_allocator_type<T>()}; 
            }

            template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                std::size_t space = alignof(T);
                return std::align(alignof(T), 1, ptr, space); 
            }
        };

        struct gpu
        {
            using id_type = int;

            static constexpr const char* name = "GPU";

            static id_type default_id() { return 0; }

            template<typename T>
            using allocator_type = std::allocator<T>;

            template<typename T>
            using aligned_allocator_type = boost::alignment::aligned_allocator_adaptor<allocator_type<T>, 64>;
            
            template<typename T>
            using vector_type = std::vector<T, aligned_allocator_type<T>>;

            /*struct handle
            {
                void wait() {}
            };*/

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                return vector_type<T>{aligned_allocator_type<T>()}; 
            }

            template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                std::size_t space = alignof(T);
                return std::align(alignof(T), 1, ptr, space); 
            }
        };

        using device_list = std::tuple<cpu,gpu>;

    } // namespace device

} // namespace gridtools


#endif /* INCLUDED_DEVICES_HPP */

