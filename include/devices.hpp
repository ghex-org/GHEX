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

#include <tuple>
#include "allocator/default_init_allocator.hpp"
#include <boost/align/aligned_allocator_adaptor.hpp>

namespace gridtools {

    namespace device {

        struct cpu
        {
            using id_type = int;

            static constexpr const char* name = "CPU";

            static id_type default_id() { return 0; }

            template<typename T>
            using allocator_type = allocator::default_init_allocator<T>;

            template<typename T>
            using aligned_allocator_type = boost::alignment::aligned_allocator_adaptor<allocator_type<T>, 64>;
            
            template<typename T>
            using vector_type = std::vector<T, aligned_allocator_type<T>>;

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                return vector_type<T>{aligned_allocator_type<T>()}; 
            }

            static void sync(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
            }

            /*template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                std::size_t space = alignof(T);
                return std::align(alignof(T), 1, ptr, space); 
            }*/
        };

#ifdef __CUDACC__
        struct gpu
        {
            using id_type = int;

            static constexpr const char* name = "GPU";

            static id_type default_id() { return 0; }

            template<typename T,typename X>
            struct vector_type_
            {
                T* m_data = nullptr;
                std::size_t m_size = 0;
                std::size_t m_capacity = 0;
                const T* data() const { return m_data; }
                T* data() { return m_data; }
                std::size_t size() const { return m_size; }
                std::size_t capacity() const { return m_capacity; }

                void resize(std::size_t new_size)
                {
                    if (new_size <= m_capacity)
                    {
                        m_size = new_size;
                    }
                    else
                    {
                        cudaFree(m_data);
                        std::size_t new_capacity = std::max(new_size, (std::size_t)(m_capacity*1.6));
                        cudaMalloc((void**)&m_data, new_capacity*sizeof(T));
                        m_capacity = new_capacity;
                        m_size = new_size;
                    }
                }

                ~vector_type_()
                {
                    if (m_capacity > 0u)
                    {
                        cudaFree(m_data);
                    }
                }

            };


            template<typename T>
            using vector_type = vector_type_<T, void>;
            
            /*struct handle
            {
                void wait() {}
            };*/

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                return {}; 
            }

            static void sync(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                cudaDeviceSynchronize();
            }

            /*template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
                static_assert(std::is_same<decltype(index),id_type>::value); // trick to prevent warnings
                std::size_t space = alignof(T);
                return std::align(alignof(T), 1, ptr, space); 
            }*/
        };

        using device_list = std::tuple<cpu,gpu>;
#else
        using device_list = std::tuple<cpu>;
#endif

    } // namespace device

} // namespace gridtools


#endif /* INCLUDED_DEVICES_HPP */

