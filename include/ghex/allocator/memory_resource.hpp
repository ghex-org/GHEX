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
#ifndef INCLUDED_GHEX_MEMORY_RESOURCE_HPP
#define INCLUDED_GHEX_MEMORY_RESOURCE_HPP

#include <experimental/memory_resource>
#include "../cuda_utils/error.hpp"

namespace gridtools {
    namespace ghex {
        namespace allocator {
                
            template<class T>
            using polymorphic_allocator = std::experimental::pmr::polymorphic_allocator<T>;

            using memory_resource = std::experimental::pmr::memory_resource;

            using std::experimental::pmr::new_delete_resource;
            using std::experimental::pmr::get_default_resource;
            
            template<typename T, std::size_t Alignment = alignof(T)>
            struct aligned_polymorphic_allocator : public polymorphic_allocator<T>
            {
                using base = polymorphic_allocator<T>;
                using base_traits = std::allocator_traits<base>;
                
                template<typename U>
                struct rebind
                {
                    using other = aligned_polymorphic_allocator<U, Alignment>;
                };

                aligned_polymorphic_allocator() noexcept = default;
                aligned_polymorphic_allocator(const aligned_polymorphic_allocator& other ) noexcept = default;

                template< class U, std::size_t N >
                aligned_polymorphic_allocator( const aligned_polymorphic_allocator<U,N>& other ) noexcept 
                : base(other.resource()) {}

                aligned_polymorphic_allocator( memory_resource* r) :  base(r) {}

                T* allocate( std::size_t n )
                {
                    return static_cast<T*>(this->resource()->allocate(n * sizeof(T), Alignment));
                }
                void deallocate(T* p, std::size_t n )
                {
                    this->resource()->deallocate(p, n * sizeof(T), Alignment);
                }
            };

#ifdef __CUDACC__
            namespace cuda {
                struct memory_resource : public std::experimental::pmr::memory_resource
                {
                    virtual void* do_allocate(std::size_t bytes, std::size_t alignment) override
                    {
                        T* ptr = nullptr;
                        GHEX_CHECK_CUDA_RESULT(cudaMalloc((void**)&ptr, bytes));
                        return ptr;
                    }
                    virtual void* do_deallocate(void* p, std::size_t bytes, std::size_t alignment) override
                    {
                        GHEX_CHECK_CUDA_RESULT(cudaFree(p));
                    }
                    virtual bool do_is_equal(const memory_resource& other) const noexcept override
                    {
                        return true;
                    }
                };
            } // namespace cuda
#endif

        } // namespace allocator
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_MEMORY_RESOURCE_HPP */


