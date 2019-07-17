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
                static_assert(std::is_same<decltype(index),id_type>::value, "trick to prevent warnings"); // trick to prevent warnings
                return vector_type<T>{aligned_allocator_type<T>()}; 
            }

            static void sync(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value, "trick to prevent warnings"); // trick to prevent warnings
            }

            static void check_error(const std::string&)
            {
            }
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

                vector_type_() = default;
                vector_type_(const vector_type_&) = delete;
                vector_type_(vector_type_&& other)
                : m_data(other.m_data), m_size(other.m_size), m_capacity(other.m_capacity)
                {
                    other.m_size = 0u;
                    other.m_capacity = 0u;
                }

                void resize(std::size_t new_size)
                {
                    if (new_size <= m_capacity)
                    {
                        m_size = new_size;
                    }
                    else
                    {
                        //cudaFree(m_data);
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
                        //int* ptr;
                        //cudaMalloc((void**)&ptr, 32*sizeof(int));
                        //cudaFree(m_data);
                        //cudaFree(ptr);
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
                static_assert(std::is_same<decltype(index),id_type>::value, "trick to prevent warnings"); // trick to prevent warnings
                return {}; 
            }

            static void sync(id_type index = default_id()) 
            { 
                static_assert(std::is_same<decltype(index),id_type>::value, "trick to prevent warnings"); // trick to prevent warnings
                cudaDeviceSynchronize();
            }

            static void check_error(const std::string& msg)
            {
                auto last_error = cudaPeekAtLastError();
                if (last_error != cudaSuccess)
                    throw std::runtime_error(msg);
            }
        };

        using device_list = std::tuple<cpu,gpu>;
#else
#ifdef GHEX_EMULATE_GPU
        struct gpu : public cpu
        {
        };

        using device_list = std::tuple<cpu,gpu>;
#else
        using device_list = std::tuple<cpu>;
#endif
#endif

    } // namespace device

} // namespace gridtools


#endif /* INCLUDED_DEVICES_HPP */

