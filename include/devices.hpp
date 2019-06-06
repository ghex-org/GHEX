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

namespace gridtools {

    namespace device {

        struct cpu
        {
            using id_type = int;

            static constexpr const char* name = "CPU";

            static id_type default_id() { return 0; }
            
            template<typename T>
            using vector_type = std::vector<T, std::allocator<T>>;

            /*struct handle
            {
                void wait() {}
            };*/

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                return vector_type<T>{std::allocator<T>()}; 
            }

            template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
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
            using vector_type = std::vector<T, std::allocator<T>>;

            /*struct handle
            {
                void wait() {}
            };*/

            template<typename T>
            static vector_type<T> make_vector(id_type index = default_id()) 
            { 
                return vector_type<T>{std::allocator<T>()}; 
            }

            template<typename T>
            static void* align(void* ptr, id_type index = default_id()) 
            {
                std::size_t space = alignof(T);
                return std::align(alignof(T), 1, ptr, space); 
            }
        };

        using device_list = std::tuple<cpu,gpu>;

    } // namespace device

} // namespace gridtools


#endif /* INCLUDED_DEVICES_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

