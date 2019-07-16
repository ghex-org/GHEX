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

#include "./communicator_base.hpp"
#include <boost/mpi/communicator.hpp>

namespace gridtools {

    namespace protocol {

        /** @brief mpi transport protocol tag */
        struct mpi {};

        /** @brief mpi communicator */
        template<>
        class communicator<mpi>
        {
        public:
            using protocol_type = mpi;
            using handle_type = boost::mpi::request;
            using address_type = int;
            template<typename T>
            using future = future_base<handle_type,T>;
            using size_type = int;

        public:
            /**
             * @brief construct from MPI_Comm object
             * @param comm MPI_Comm communicator
             */
            communicator(const MPI_Comm& comm)
            :   m_comm(comm, boost::mpi::comm_attach) {}

            /** @brief copy construct */
            communicator(const communicator& other) 
            : communicator(other.m_comm) {} 

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
            future<void> isend(address_type dest, int tag, const T* buffer, int n) const
            {
                return {std::move(m_comm.isend(dest, tag, reinterpret_cast<const char*>(buffer), sizeof(T)*n))};
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
            future<void> irecv(address_type source, int tag, T* buffer, int n) const
            {
                return {std::move(m_comm.irecv(source, tag, reinterpret_cast<char*>(buffer), sizeof(T)*n))};
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
            template<typename T, template<typename, typename> /*typename*/class Vector, typename... Args> 
            future<void> isend(address_type dest, int tag, const Vector<T,Args...>& vec) const
            {
                return {std::move(m_comm.isend(dest, tag, reinterpret_cast<const char*>(vec.data()), sizeof(T)*vec.size()))};
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
            template<typename T, template<typename, typename> /*typename*/class Vector = std::vector, typename Allocator = std::allocator<T>> 
            [[nodiscard]] future<Vector<T,Allocator>> irecv(address_type source, int tag, int n, const Allocator& a = Allocator()) const
            {
                using vector_type = Vector<T,Allocator>;
                using size_type   = typename vector_type::size_type;
                vector_type vec( size_type(n), a );
                return { std::move(vec), std::move(m_comm.irecv(source, tag, reinterpret_cast<char*>(vec.data()), sizeof(T)*n)) };
            }

        private:
            boost::mpi::communicator m_comm;
        };

    } // namespace protocol

} // namespace gridtools

#endif /* INCLUDED_MPI_HPP */

