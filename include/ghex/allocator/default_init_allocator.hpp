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
#ifndef INCLUDED_GHEX_ALLOCATOR_DEFAULT_INIT_ALLOCATOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_DEFAULT_INIT_ALLOCATOR_HPP

#include <memory>

namespace gridtools {

    namespace ghex {

        namespace allocator {

            /** @brief Allocator adaptor that interposes construct() calls to convert value initialization 
             * into default initialization.
             * @tparam T type to allocate
             * @tparam A base allocator*/
            template <typename T, typename A=std::allocator<T>>
            class default_init_allocator : public A 
            {
            private: // member types
                using a_t = std::allocator_traits<A>;

            public: // member types
                template <typename U> struct rebind 
                {
                    using other = default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
                };

            public: // constructor
                using A::A;

            public: // member functions
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

        } // namespace allocator

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_DEFAULT_INIT_ALLOCATOR_HPP */

