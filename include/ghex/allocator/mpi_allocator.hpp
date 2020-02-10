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
#ifndef INCLUDED_GHEX_ALLOCATOR_MPI_ALLOCATOR_HPP
#define INCLUDED_GHEX_ALLOCATOR_MPI_ALLOCATOR_HPP

#include <mpi.h>

namespace gridtools {

    namespace ghex {

        namespace allocator {

            template <class T>
            struct mpi_allocator {
                
                typedef T value_type;

                mpi_allocator() = default;
                
                template <class U> constexpr mpi_allocator(const mpi_allocator<U>&) noexcept {}
                
                [[nodiscard]] T* allocate(std::size_t n) 
                {
                    if(n > std::size_t(-1) / sizeof(T)) throw std::bad_alloc();
                    void* baseptr;
                    if (MPI_SUCCESS == MPI_Alloc_mem(n*sizeof(T), MPI_INFO_NULL, &baseptr)) 
                        return static_cast<T*>(baseptr);
                    throw std::bad_alloc();
                }

                void deallocate(T* p, std::size_t) noexcept 
                { 
                    MPI_Free_mem(p);
                }

                template <typename U>
                void construct(U* ptr) noexcept(std::is_nothrow_default_constructible<U>::value) 
                {
                    ::new(static_cast<void*>(ptr)) U;
                }
              
                template <typename U, typename...Args>
                void construct(U* ptr, Args&&... args) 
                {
                    ::new (static_cast<void*>(ptr)) U(std::forward<Args>(args)...);
                }
            };
            template <class T, class U>
            bool operator==(const mpi_allocator<T>&, const mpi_allocator<U>&) { return true; }
            template <class T, class U>
            bool operator!=(const mpi_allocator<T>&, const mpi_allocator<U>&) { return false; }

        } // namespace allocator

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_ALLOCATOR_MPI_ALLOCATOR_HPP */

