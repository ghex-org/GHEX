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
#ifndef INCLUDED_SETUP_HPP
#define INCLUDED_SETUP_HPP

#include "./communicator_base.hpp"
#include "./mpi_comm.hpp"
#include <vector>

namespace gridtools {

    namespace protocol {

        /** @brief special mpi communicator used for setup phase */
        class setup_communicator
        {
        public:
            using handle_type = ::gridtools::ghex::mpi::request;
            using address_type = int;
            template<typename T>
            using future = future_base<handle_type,T>;

        private:
            ::gridtools::ghex::mpi::mpi_comm m_comm;

        public:
            setup_communicator(const MPI_Comm& comm)
            :   m_comm(comm) {}

            setup_communicator(const setup_communicator& other) 
            : setup_communicator(other.m_comm) {} 

            address_type address() const { return m_comm.rank(); }

            address_type rank() const { return m_comm.rank(); }

            void barrier() { m_comm.barrier(); }

            template<typename T>
            void send(int dest, int tag, const T & value)
            {
                GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(&value), sizeof(T), MPI_BYTE, dest, tag, m_comm));
            }

            template<typename T>
            ::gridtools::ghex::mpi::status recv(int source, int tag, T & value)
            {
                MPI_Status status;
                GHEX_CHECK_MPI_RESULT(MPI_Recv(reinterpret_cast<void*>(&value), sizeof(T), MPI_BYTE, source, tag, m_comm, &status));
                return {status};
            }

            template<typename T>
            void send(int dest, int tag, const T* values, int n)
            {
                GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(values), sizeof(T)*n, MPI_BYTE, dest, tag, m_comm));
            }

            template<typename T>
            ::gridtools::ghex::mpi::status recv(int source, int tag, T* values, int n)
            {
                MPI_Status status;
                GHEX_CHECK_MPI_RESULT(MPI_Recv(reinterpret_cast<void*>(values), sizeof(T)*n, MPI_BYTE, source, tag, m_comm, &status));
                return {status};
            }

            template<typename T> 
            void broadcast(T& value, int root)
            {
                GHEX_CHECK_MPI_RESULT(MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, m_comm));
            }

            template<typename T> 
            void broadcast(T * values, int n, int root)
            {
                GHEX_CHECK_MPI_RESULT(MPI_Bcast(values, sizeof(T)*n, MPI_BYTE, root, m_comm));
            }

            template<typename T>
            future< std::vector<std::vector<T>> > all_gather(const std::vector<T>& payload, const std::vector<int>& sizes)
            {
                handle_type h;
                std::vector<int> displs(m_comm.size());
                std::vector<int> recvcounts(m_comm.size());
                std::vector<std::vector<T>> res(m_comm.size());
                for (int i=0; i<m_comm.size(); ++i)
                {
                    res[i].resize(sizes[i]);
                    recvcounts[i] = sizes[i]*sizeof(T);
                    displs[i] = reinterpret_cast<char*>(&res[i][0]) - reinterpret_cast<char*>(&res[0][0]);
                }
                GHEX_CHECK_MPI_RESULT( MPI_Iallgatherv(
                    &payload[0], payload.size()*sizeof(T), MPI_BYTE,
                    &res[0][0], &recvcounts[0], &displs[0], MPI_BYTE,
                    m_comm, 
                    &h.m_req));
                return {std::move(res), std::move(h)};
            }

            template<typename T>
            future< std::vector<T> > all_gather(const T& payload)
            {
                std::vector<T> res(m_comm.size());
                handle_type h;
                GHEX_CHECK_MPI_RESULT(
                    MPI_Iallgather
                    (&payload, sizeof(T), MPI_BYTE,
                    &res[0], sizeof(T), MPI_BYTE,
                    m_comm,
                    &h.m_req));
                return {std::move(res), std::move(h)};
            } 

        };

    } // namespace protocol

} // namespace gridtools

#endif /* INCLUDED_SETUP_HPP */

