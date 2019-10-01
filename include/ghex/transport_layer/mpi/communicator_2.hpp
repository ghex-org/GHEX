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
#ifndef INCLUDED_MPI_HPP
#define INCLUDED_MPI_HPP

#include "../communicator.hpp"
#include "./communicator_base.hpp"
#include "./future.hpp"

namespace gridtools {
    
    namespace ghex {

        namespace tl {

            /** @brief mpi transport protocol tag */
            struct mpi_tag {};

            /** @brief mpi communicator */
            template<>
            class communicator<mpi_tag>
            {
            public:
                using protocol_type = mpi_tag;
                //using handle_type = ::gridtools::ghex::mpi::request;
                using handle_type = ::gridtools::ghex::tl::mpi::request;
                using address_type = int;
                template<typename T>
                using future = ::gridtools::ghex::tl::mpi::future<T>; //future_base<handle_type,T>;
                using size_type = int;

            //private:
                //::gridtools::ghex::mpi::mpi_comm m_comm;
                ::gridtools::ghex::tl::mpi::communicator_base m_comm;

                operator const MPI_Comm&() const noexcept { return m_comm; }
                operator       MPI_Comm&()       noexcept { return m_comm; }

            public:
                ///**
                // * @brief construct from MPI_Comm object
                // * @param comm MPI_Comm communicator
                // */
                //communicator(const MPI_Comm& comm)
                //:   m_comm(comm) {}

                ///** @brief copy construct */
                //communicator(const communicator& other) = delete; 
                //: communicator(other.m_comm) {}

                /** @return address of this process */
                address_type address() const { return m_comm.rank(); }
                
                /** @return rank of this process */
                address_type rank() const { return m_comm.rank(); }

                /** @return size of communicator group*/
                size_type size() const { return m_comm.size(); }

                void barrier() { m_comm.barrier(); }

                /**
                 * @brief non-blocking send
                 * @tparam T data type
                 * @param dest destination rank
                 * @param tag message tag
                 * @param buffer pointer to source buffer
                 * @param n number of elements in buffer
                 * @return completion handle
                 */
                template<typename T>
                [[nodiscard]] future<void> isend(address_type dest, int tag, const T* buffer, int n) const
                {
                    handle_type req;
                    GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(buffer),sizeof(T)*n, MPI_BYTE, dest, tag, m_comm, &req.get()));
                    return req;
                }

                /**
                 * @brief non-blocking receive
                 * @tparam T data type
                 * @param source source rank
                 * @param tag message tag
                 * @param buffer pointer destination buffer
                 * @param n number of elements in buffer
                 * @return completion handle
                 */
                template<typename T>
                [[nodiscard]] future<void> irecv(address_type source, int tag, T* buffer, int n) const
                {
                    handle_type req;
                    GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(buffer),sizeof(T)*n, MPI_BYTE, source, tag, m_comm, &req.get()));
                    return req;
                }

                /**
                 * @brief non-blocking send (vector interface)
                 * @tparam T data type
                 * @tparam Vector vector type (contiguous memory)
                 * @tparam Allocator allocator type
                 * @param dest destination rank
                 * @param tag message tag
                 * @param vec source vector
                 * @return completion handle
                 */
                template<typename T, template<typename, typename> class Vector, typename Allocator> 
                [[nodiscard]] future<void> isend(address_type dest, int tag, const Vector<T,Allocator>& vec) const
                {
                    handle_type req;
                    GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(vec.data()),sizeof(T)*vec.size(), MPI_BYTE, dest, tag, m_comm, &req.get()));
                    return req;
                }

                /**
                 * @brief non-blocking receive (vector interface)
                 * @tparam T data type
                 * @tparam Vector vector type (contiguous memory)
                 * @tparam Allocator allocator type
                 * @param source source rank
                 * @param tag message tag
                 * @param n number of elements to receive
                 * @param a allocator instance
                 * @return future with vector of data
                 */
                template<typename T, template<typename, typename> class Vector = std::vector, typename Allocator = std::allocator<T>> 
                [[nodiscard]] future<Vector<T,Allocator>> irecv(address_type source, int tag, int n, const Allocator& a = Allocator()) const
                {
                    using vector_type = Vector<T,Allocator>;
                    using size_type   = typename vector_type::size_type;
                    vector_type vec( size_type(n), a );
                    handle_type req;
                    GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(vec.data()),sizeof(T)*n, MPI_BYTE, source, tag, m_comm, &req.get()));
                    return { vec, req };
                }
            };

        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_MPI_HPP */

