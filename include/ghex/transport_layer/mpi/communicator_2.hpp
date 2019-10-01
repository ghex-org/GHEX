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
#include "./communicator_traits.hpp"

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
                using base_type     = mpi::communicator_base;
                using address_type  = typename base_type::rank_type;
                using rank_type     = typename base_type::rank_type;
                using size_type     = typename base_type::size_type;
                using tag_type      = typename base_type::tag_type;
                using request       = mpi::request;
                using status        = mpi::status;
                template<typename T>
                using future        = mpi::future<T>;
                using traits        = mpi::communicator_traits;

            //private:
                base_type m_comm;

                operator const MPI_Comm&() const noexcept { return m_comm; }
                operator       MPI_Comm&()       noexcept { return m_comm; }

            public:

                communicator() = default;
                communicator(const base_type& c) : m_comm{c} {}
                communicator(const MPI_Comm& c) : m_comm{c} {}
                communicator(const traits& t = traits{}) : m_comm{t.communicator()} {}
                
                communicator(const communicator&) = default;
                communicator(communicator&&) = default;

                communicator& operator=(const communicator&) = default;
                communicator& operator=(communicator&&) = default;

                /** @return address of this process */
                address_type address() const { return m_comm.rank(); }
                
                /** @return rank of this process */
                rank_type rank() const { return m_comm.rank(); }

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
                [[nodiscard]] future<void> send(rank_type dest, tag_type tag, const T* buffer, int n) const
                {
                    request req;
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
                [[nodiscard]] future<void> recv(rank_type source, tag_type tag, T* buffer, int n) const
                {
                    request req;
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
                [[nodiscard]] future<void> send(rank_type dest, tag_type tag, const Vector<T,Allocator>& vec) const
                {
                    request req;
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
                [[nodiscard]] future<Vector<T,Allocator>> recv(rank_type source, tag_type tag, int n, const Allocator& a = Allocator()) const
                {
                    using vector_type = Vector<T,Allocator>;
                    using size_type   = typename vector_type::size_type;
                    vector_type vec( size_type(n), a );
                    request req;
                    GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(vec.data()),sizeof(T)*n, MPI_BYTE, source, tag, m_comm, &req.get()));
                    return { vec, req };
                }
            };

        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_MPI_HPP */

