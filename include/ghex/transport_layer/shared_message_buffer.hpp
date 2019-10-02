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
#ifndef INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP
#define INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP

#include <memory>
#include "./message_buffer.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename Allocator = std::allocator<unsigned char>>
            class shared_message_buffer
            {
            public: // member types

                using message_type      = message_buffer<Allocator>;
                using byte              = typename message_type::byte;
                using value_type        = typename message_type::value_type;
                using allocation_type   = typename message_type::allocation_type;
                using allocator_type    = typename message_type::allocator_type;
                using pointer           = typename message_type::pointer;
                using const_pointer     = typename message_type::const_pointer;
                using raw_pointer       = typename message_type::raw_pointer;
                using raw_const_pointer = typename message_type::raw_const_pointer;
    
                static constexpr bool can_be_shared = true;

            private: // members

                std::shared_ptr<message_type> m_message;

            public: // ctors

                template<typename Alloc = Allocator>
                shared_message_buffer(Alloc alloc = Alloc{})
                : m_message{std::make_shared<message_type>(alloc)}
                {}
    
                template<typename Alloc = Allocator>
                shared_message_buffer(size_t size_, Alloc alloc = Alloc{})
                : m_message{std::make_shared<message_type>(size_,alloc)}
                {}

                shared_message_buffer(message_type&& m)
                : m_message{std::make_shared<message_type>(std::move(m))}
                {}

                shared_message_buffer(const shared_message_buffer&) = default;
                shared_message_buffer(shared_message_buffer&&) = default;
                shared_message_buffer& operator=(const shared_message_buffer&) = default;
                shared_message_buffer& operator=(shared_message_buffer&&) = default;
            
            public: // member functions
    
                bool is_shared() const { return use_count() > 1; }
                auto use_count() const { return m_message.use_count(); }

                std::size_t size() const noexcept { return m_message->size(); }
                std::size_t capacity() const noexcept { return m_message->capacity(); }

                raw_const_pointer data() const noexcept { return m_message->data(); }
                raw_pointer data() noexcept { return m_message->data(); }

                template <typename T>
                T* data() { return m_message->template data<T>(); }
                template <typename T>
                const T* data() const { return m_message->template data<T>(); }

                raw_const_pointer begin() const noexcept { return m_message->begin(); }
                raw_const_pointer end() const noexcept { return m_message->end(); }
                raw_pointer begin() noexcept { return m_message->begin(); }
                raw_pointer end() noexcept { return m_message->end(); }

                void reserve(std::size_t n) { m_message->reserve(n); }
                void resize(std::size_t n) { m_message->resize(n); }
                void clear() { m_message->clear(); }
                void swap(shared_message_buffer& other) { std::swap(m_message, other.m_message); }
            };

            template<typename A>
            void swap(shared_message_buffer<A>& a, shared_message_buffer<A>& b)
            {
                a.swap(b);
            }

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_SHARED_MESSAGE_BUFFER_HPP */

