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
#ifndef INCLUDED_GHEX_TL_MESSAGE_BUFFER_HPP
#define INCLUDED_GHEX_TL_MESSAGE_BUFFER_HPP

#include <cassert>
#include "../allocator/allocation.hpp"
#include "../common/to_address.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename Allocator = std::allocator<unsigned char>>
            class message_buffer
            {
            public: // member types

                using byte              = unsigned char;
                using value_type        = byte;
                using allocation_type   = ::gridtools::ghex::allocator::allocation<Allocator,byte>;
                using allocator_type    = typename allocation_type::alloc_type;
                using pointer           = typename allocation_type::pointer;
                using const_pointer     = typename allocation_type::const_pointer;
                using raw_pointer       = byte*;
                using raw_const_pointer = const byte*;

                static constexpr bool can_be_shared = false;

            private: // members
                
                allocation_type m_allocation;
                std::size_t m_size = 0u;

            public: // ctors
    
                //template<typename Alloc = Allocator>
                message_buffer(Allocator alloc = Allocator{})
                : m_allocation( alloc )
                { }
    
                template<typename Alloc = Allocator>
                message_buffer(size_t size_, Alloc alloc = Alloc{})
                : m_allocation( alloc, size_ )
                , m_size{size_}
                {}

                message_buffer(message_buffer&& other)
                : m_allocation{std::move(other.m_allocation)}
                , m_size{other.m_size}
                {
                    other.m_size = 0;
                }

                message_buffer& operator=(message_buffer& other)
                {
                    m_allocation = std::move(other.m_allocation);
                    m_size = other.m_size;
                    other.m_size = 0u;
                    return *this;
                }

                message_buffer(const message_buffer&) = delete;
                message_buffer& operator=(const message_buffer&) = delete;

            public: // member functions
    
                bool is_shared() const { return can_be_shared; }
                std::size_t use_count() const { return 1; }

                std::size_t size() const noexcept { return m_size; }
                std::size_t capacity() const noexcept { return m_allocation.m_capacity; }

                raw_const_pointer data() const noexcept { return ::gridtools::ghex::to_address(m_allocation.m_pointer); }
                raw_pointer data() noexcept { return ::gridtools::ghex::to_address(m_allocation.m_pointer); }

                template <typename T>
                T* data() noexcept
                {
                    raw_pointer byte_ptr = data();
                    assert(reinterpret_cast<std::uintptr_t>(byte_ptr) % alignof(T) == 0);
                    return reinterpret_cast<T*>(byte_ptr);
                }
                template <typename T>
                const T* data() const noexcept
                {
                    raw_const_pointer byte_ptr = data();
                    assert(reinterpret_cast<std::uintptr_t>(byte_ptr) % alignof(T) == 0);
                    return reinterpret_cast<const T*>(byte_ptr);
                }

                raw_const_pointer begin() const noexcept { return data(); }
                raw_const_pointer end() const noexcept { return data()+m_size; }
                raw_pointer begin() noexcept { return data(); }
                raw_pointer end() noexcept { return data()+m_size; }

                void reserve(std::size_t n)
                {
                    if (n<=m_allocation.m_capacity) return;
                    allocation_type new_allocation(m_allocation.m_alloc, n);
                    void* ptr = &m_allocation;
                    m_allocation.~allocation_type();
                    new(ptr) allocation_type(std::move(new_allocation));
                }

                void resize(std::size_t n)
                {
                    if (n==m_size) return;
                    if (n==0)
                    {
                        m_size = 0;
                        return;
                    }
                    if (n < m_size)
                    {
                        m_size = n;
                        return;
                    }
                    reserve(n);
                    m_size = n;
                }

                void clear() { resize(0); }

                void swap(message_buffer& other)
                {
                    m_allocation.swap(other.m_allocation);
                    const auto s = m_size;
                    m_size = other.m_size;
                    other.m_size = s;
                }
            };

            template<typename A>
            void swap(message_buffer<A>& a, message_buffer<A>& b)
            {
                a.swap(b);
            }

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif/*INCLUDED_GHEX_TL_MESSAGE_BUFFER_HPP */

