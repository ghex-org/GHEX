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
#ifndef INCLUDED_GHEX_ALLOCATOR_ALIGNED_ALLOCATOR_ADAPTOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_ALIGNED_ALLOCATOR_ADAPTOR_HPP

#include <type_traits>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "../common/to_address.hpp"

namespace gridtools {
    namespace ghex {
        namespace allocator {

            template<typename Allocator, std::size_t Alignment, typename Enable = void>
            struct aligned_allocator_adaptor
            : public std::allocator_traits<Allocator>::template rebind_alloc<unsigned char>
            {
            public: // static constants

                static constexpr std::uintptr_t mask = ~(Alignment-1u);
                static constexpr std::uintptr_t offset = alignof(std::max_align_t) + (Alignment-alignof(std::max_align_t));
            
            public: // member types

                using base               = Allocator;
                using base_traits        = std::allocator_traits<base>;
                
                using byte               = unsigned char;
                using base_byte          = typename base_traits::template rebind_alloc<byte>;
                using base_byte_traits   = typename std::allocator_traits<base_byte>;
        
                using pointer            = typename base_traits::pointer;
                using const_pointer      = typename base_traits::const_pointer;
                using void_pointer       = typename base_traits::void_pointer;
                using const_void_pointer = typename base_traits::const_void_pointer;
                using value_type         = typename base::value_type;
                using size_type          = typename base_traits::size_type;
                using difference_type    = typename base_traits::difference_type;

                using pointer_traits     = std::pointer_traits<pointer>;

                using propagate_on_container_copy_assignment = typename base_byte_traits::propagate_on_container_copy_assignment;
                using propagate_on_container_move_assignment = typename base_byte_traits::propagate_on_container_move_assignment;
                using propagate_on_container_swap            = typename base_byte_traits::propagate_on_container_swap;

                template<typename U>
                struct rebind
                {
                    using other = aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>, Alignment>;
                };

            public: // ctors

                template<typename Alloc = Allocator, typename std::enable_if<std::is_default_constructible<Alloc>::value, int>::type=0>
                aligned_allocator_adaptor() : base_byte()
                {
                    static_assert(std::is_same<Alloc, Allocator>::value, "this is not a function template");
                }
                aligned_allocator_adaptor(const Allocator& alloc) : base_byte{alloc} {}
                aligned_allocator_adaptor(const aligned_allocator_adaptor&) = default;
                aligned_allocator_adaptor(Allocator&& alloc) : base_byte{std::move(alloc)} {}
                aligned_allocator_adaptor(aligned_allocator_adaptor&&) = default;

                template<typename U>
                aligned_allocator_adaptor(const typename base_traits::template rebind_alloc<U>& alloc) : base_byte{alloc} {}
                template<typename U>
                aligned_allocator_adaptor(typename base_traits::template rebind_alloc<U>&& alloc) : base_byte{std::move(alloc)} {}

                template<typename U>
                aligned_allocator_adaptor(const aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>,Alignment>& alloc) 
                : base_byte{ static_cast<typename std::remove_reference_t<decltype(alloc)>::base_byte>(alloc) } {}
                
                template<typename U>
                aligned_allocator_adaptor(aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>,Alignment>&& alloc) 
                : base_byte{ static_cast<typename std::remove_reference_t<decltype(alloc)>::base_byte>(alloc) } {}
                
            public: // member functions

                inline pointer allocate(size_type n)
                {
                    auto vptr = base_byte_traits::allocate(*this, n*sizeof(value_type) + offset);
                    void* vvptr = ::gridtools::ghex::to_address(vptr);
                    void* res = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(vvptr)+offset) & mask);
                    *((void**)res - 1) = vvptr;
                    return pointer_traits::pointer_to(*reinterpret_cast<value_type*>(res));
                }

                inline pointer allocate(size_type n, const_void_pointer cvptr)
                {
                    auto vptr = base_byte_traits::allocate(*this, n*sizeof(value_type) + offset, cvptr);
                    void* vvptr = ::gridtools::ghex::to_address(vptr);
                    void* res = reinterpret_cast<void*>((reinterpret_cast<std::uintptr_t>(vvptr)+offset) & mask);
                    *((void**)res - 1) = vvptr;
                    return pointer_traits::pointer_to(*reinterpret_cast<value_type*>(res));
                }

                inline void deallocate(pointer ptr, size_type n)
                {
                    void* vvptr = ::gridtools::ghex::to_address(ptr);
                    typename base_byte_traits::pointer bptr = reinterpret_cast<byte*>(*((void**)vvptr - 1));
                    base_byte_traits::deallocate(*this, bptr, n*sizeof(value_type) + offset);
                }

                inline size_type max_size() const noexcept 
                { 
                    return base_byte_traits::max_size( *this ); 
                }

                inline aligned_allocator_adaptor select_on_container_copy_construction() const
                { 
                    return base_byte_traits::select_on_container_copy_construction( *this ); 
                }

                void swap(aligned_allocator_adaptor& other)
                {
                    std::swap(static_cast<base_byte&>(*this), static_cast<base_byte&>(other));
                }

                bool operator==(const aligned_allocator_adaptor& other) const
                {
                    return (static_cast<const base_byte&>(*this) == static_cast<const base_byte&>(other));
                }

                bool operator!=(const aligned_allocator_adaptor& other) const
                {
                    return !(operator==(other));
                }
            };

            // fallback to normal Allocator if Alignment is small enough
            template<typename Allocator, std::size_t Alignment>
            struct aligned_allocator_adaptor<Allocator, Alignment, typename std::enable_if<(Alignment<=alignof(std::max_align_t))>::type>
            : public Allocator
            {
                static constexpr std::uintptr_t offset = 0;
            
            public: // member types

                using base               = Allocator;
                using base_traits        = std::allocator_traits<base>;
        
                using pointer            = typename base_traits::pointer;
                using const_pointer      = typename base_traits::const_pointer;
                using void_pointer       = typename base_traits::void_pointer;
                using const_void_pointer = typename base_traits::const_void_pointer;
                using value_type         = typename base::value_type;
                using size_type          = typename base_traits::size_type;
                using difference_type    = typename base_traits::difference_type;

                using pointer_traits     = std::pointer_traits<pointer>;

                using propagate_on_container_copy_assignment = typename base_traits::propagate_on_container_copy_assignment;
                using propagate_on_container_move_assignment = typename base_traits::propagate_on_container_move_assignment;
                using propagate_on_container_swap            = typename base_traits::propagate_on_container_swap;

                template<typename U>
                struct rebind
                {
                    using other = aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>, Alignment>;
                };

            public: // ctors

                template<typename Alloc = Allocator, typename std::enable_if<std::is_default_constructible<Alloc>::value, int>::type=0>
                aligned_allocator_adaptor() : base()
                {
                    static_assert(std::is_same<Alloc, Allocator>::value, "this is not a function template");
                }
                aligned_allocator_adaptor(const Allocator& alloc) : base{alloc} {}
                aligned_allocator_adaptor(const aligned_allocator_adaptor&) = default;
                aligned_allocator_adaptor(Allocator&& alloc) : base{std::move(alloc)} {}
                aligned_allocator_adaptor(aligned_allocator_adaptor&&) = default;

                template<typename U>
                aligned_allocator_adaptor(const typename base_traits::template rebind_alloc<U>& alloc) : base{alloc} {}
                template<typename U>
                aligned_allocator_adaptor(typename base_traits::template rebind_alloc<U>&& alloc) : base{std::move(alloc)} {}

                template<typename U>
                aligned_allocator_adaptor(const aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>,Alignment>& alloc) 
                : base{ static_cast<typename std::remove_reference_t<decltype(alloc)>::base>(alloc) } {}
                
                template<typename U>
                aligned_allocator_adaptor(aligned_allocator_adaptor<typename base_traits::template rebind_alloc<U>,Alignment>&& alloc) 
                : base{ static_cast<typename std::remove_reference_t<decltype(alloc)>::base>(alloc) } {}

                inline aligned_allocator_adaptor select_on_container_copy_construction() const
                { 
                    return base_traits::select_on_container_copy_construction( *this ); 
                }
            };

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_ALIGNED_ALLOCATOR_ADAPTOR_HPP */

