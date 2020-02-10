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
#ifndef INCLUDED_GHEX_ALLOCATOR_POOL_ALLOCATOR_ADAPTOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_POOL_ALLOCATOR_ADAPTOR_HPP

#include <memory>
#include <vector>
#include <unordered_map>
#include "../common/to_address.hpp"

namespace gridtools {
    namespace ghex {
        namespace allocator {

            template<typename Allocator>
            struct pool_impl
            {
                using byte               = unsigned char;
                using alloc_t            = typename std::allocator_traits<Allocator>::template rebind_alloc<byte>;
                using traits             = std::allocator_traits<alloc_t>;
                using pointer            = typename traits::pointer;
                using const_void_pointer = typename traits::const_void_pointer;
                using size_type          = typename traits::size_type;
                using pointer_traits     = std::pointer_traits<pointer>;
                using key_type           = std::size_t;
                using mapped_type        = std::vector<byte*>;
                using map_type           = std::unordered_map<key_type, mapped_type>;

                static_assert(std::is_same<alloc_t, Allocator>::value, "must be a byte allocator");

                alloc_t m_alloc;
                map_type m_map;

                pool_impl(Allocator alloc)
                : m_alloc{alloc}
                {}

                pool_impl(const pool_impl&) = delete;
                pool_impl& operator=(const pool_impl&) = delete;

                pool_impl(pool_impl&&) = default;
                pool_impl& operator=(pool_impl&&) = default;

                ~pool_impl()
                {
                    for (const auto& kvp : m_map)
                        for (auto ptr : kvp.second)
                            traits::deallocate(m_alloc, pointer_traits::pointer_to(*ptr), kvp.first);
                }

                pointer allocate(size_type n, const_void_pointer cvptr = nullptr)
                {
                    auto x = m_map.find(n);
                    if (x == m_map.end()) return traits::allocate(m_alloc, n, cvptr);
                    auto& kvp = *x;
                    if (kvp.second.size() == 0u) return traits::allocate(m_alloc, n, cvptr);
                    byte* ptr = kvp.second.back();
                    kvp.second.pop_back();
                    return pointer_traits::pointer_to(*ptr);
                }

                void deallocate(pointer ptr, size_type n)
                {
                    byte* bptr = ::gridtools::ghex::to_address(ptr);
                    m_map[n].push_back(bptr);
                }
            };

            template<typename Allocator>
            struct pool_allocator_adaptor
            : public Allocator
            {
            public: // member types

                using base               = Allocator;
                using base_traits        = std::allocator_traits<Allocator>;
                using pointer            = typename base_traits::pointer;
                using const_pointer      = typename base_traits::const_pointer;
                using void_pointer       = typename base_traits::void_pointer;
                using const_void_pointer = typename base_traits::const_void_pointer;
                using value_type         = typename base::value_type;
                using size_type          = typename base_traits::size_type;
                using difference_type    = typename base_traits::difference_type;
                using pointer_traits     = std::pointer_traits<pointer>;

                using byte               = unsigned char;
                using byte_base          = typename base_traits::template rebind_alloc<byte>;
                using byte_base_traits   = std::allocator_traits<byte_base>;
                using byte_pointer_traits     = std::pointer_traits<typename byte_base_traits::pointer>;

                template<typename U>
                struct rebind
                {
                    using other = pool_allocator_adaptor<typename base_traits::template rebind_alloc<U>>;
                };

            public: // members

                pool_impl<byte_base>* m_pool;

            public: // ctors

                template<typename Alloc = Allocator, typename std::enable_if<std::is_default_constructible<Alloc>::value, int>::type=0>
                pool_allocator_adaptor(pool_impl<byte_base>* p)
                : base()
                , m_pool{ p }
                {
                    static_assert(std::is_same<Alloc, Allocator>::value, "this is not a function template");
                }
                pool_allocator_adaptor(pool_impl<byte_base>* p, Allocator alloc)
                : base(alloc)
                , m_pool{ p }
                {}

                pool_allocator_adaptor(const pool_allocator_adaptor&) = default;
                pool_allocator_adaptor(pool_allocator_adaptor&&) = default;
                pool_allocator_adaptor& operator=(const pool_allocator_adaptor&) = default;
                pool_allocator_adaptor& operator=(pool_allocator_adaptor&&) = default;

                template<typename U>
                pool_allocator_adaptor(const typename rebind<U>::other& other)
                : base( static_cast< typename base_traits::template rebind_alloc<U> >(other) )
                , m_pool{ other.m_pool }
                {}
                template<typename U>
                pool_allocator_adaptor(typename rebind<U>::other&& other)
                : base( static_cast< typename base_traits::template rebind_alloc<U> >(other) )
                , m_pool{ other.m_pool }
                {}

                void swap(pool_allocator_adaptor& other)
                {
                    std::swap(m_pool, other.m_pool);
                    std::swap(static_cast<base&>(*this), static_cast<base&>(other));
                }

            public: // allocate, deallocate

                pointer allocate(size_type n, const_void_pointer cvptr = nullptr)
                {
                    return
                    pointer_traits::pointer_to(
                        *reinterpret_cast<value_type*>(
                            ::gridtools::ghex::to_address(
                                m_pool->allocate(n*sizeof(value_type), cvptr)
                            )
                        )
                    );
                }

                void deallocate(pointer ptr, size_type n)
                {
                    auto bptr = byte_pointer_traits::pointer_to(
                        *reinterpret_cast<byte*>(
                            ::gridtools::ghex::to_address(ptr)
                        )
                    );
                    m_pool->deallocate(bptr, n*sizeof(value_type));
                }

            public: // container hooks
                
                using propagate_on_container_copy_assignment = std::true_type;
                using propagate_on_container_move_assignment = std::true_type;
                using propagate_on_container_swap            = std::true_type;

                pool_allocator_adaptor select_on_container_copy_construction() const
                {
                    return *this;
                }

            public: // comparison

                using is_always_equal = std::false_type;

                friend bool operator==(const pool_allocator_adaptor& a, const pool_allocator_adaptor& b)
                {
                    return a.m_pool == b.m_pool;
                }

                friend bool operator!=(const pool_allocator_adaptor& a, const pool_allocator_adaptor& b)
                {
                    return a.m_pool != b.m_pool;
                }

            public: // other member functions

                inline size_type max_size() const noexcept 
                { 
                    return base_traits::max_size(*this); 
                }
            };


            template<typename BasicAllocator>
            struct pool
            {
                using allocator_type = pool_allocator_adaptor<BasicAllocator>;
                using byte_base      = typename allocator_type::byte_base;

                std::unique_ptr<pool_impl<byte_base>> m_pool_impl;

                pool(BasicAllocator alloc)
                : m_pool_impl( new pool_impl<byte_base>{alloc} )
                {}

                pool(const pool&) = delete;
                pool(pool&&) = default;

                allocator_type get_allocator() const
                {
                    return { m_pool_impl.get() };
                }
            };

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_POOL_ALLOCATOR_ADAPTOR_HPP */

