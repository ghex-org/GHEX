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
#include "./future.hpp"
#include <vector>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

            /** @brief special mpi communicator used for setup phase */
            class setup_communicator
            {
            public:
                using handle_type = request;
                using address_type = communicator_base::rank_type;
                template<typename T>
                using future = future<T>;

            private:
                communicator_base m_comm;

            public:
                setup_communicator(const MPI_Comm& comm) noexcept
                :   m_comm{comm} {}

                setup_communicator(const setup_communicator&) = delete;
                setup_communicator& operator=(const setup_communicator&) = delete;
                setup_communicator(setup_communicator&&) noexcept = default;
                setup_communicator& operator=(setup_communicator&&) noexcept = default;

                address_type address() const { return m_comm.rank(); }

                address_type rank() const { return m_comm.rank(); }

                void barrier() { m_comm.barrier(); }

                template<typename T>
                void send(int dest, int tag, const T & value)
                {
                    GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(&value), sizeof(T), MPI_BYTE, dest, tag, m_comm));
                }

                template<typename T>
                status recv(int source, int tag, T & value)
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
                status recv(int source, int tag, T* values, int n)
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
                    std::vector<std::vector<T>> res(m_comm.size());
                    for (int neigh=0; neigh<m_comm.size(); ++neigh)
                    {
                        res[neigh].resize(sizes[neigh]);
                    }
                    std::vector<handle_type> m_reqs;
                    m_reqs.reserve((m_comm.size())*2);
                    for (int neigh=0; neigh<m_comm.size(); ++neigh)
                    {
                        m_reqs.push_back( handle_type{} );
                        GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(res[neigh].data()), sizeof(T)*sizes[neigh], MPI_BYTE, neigh, 99, m_comm, &m_reqs.back().get()));
                    }
                    for (int neigh=0; neigh<m_comm.size(); ++neigh)
                    {
                        m_reqs.push_back( handle_type{} );
                        GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(payload.data()), sizeof(T)*payload.size(), MPI_BYTE, neigh, 99, m_comm, &m_reqs.back().get()));
                    }
                    for (auto& r : m_reqs)
                        r.wait();

                    return {std::move(res), std::move(m_reqs.front())};

                    /*handle_type h;
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
                    return {std::move(res), std::move(h)};*/
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
                        &h.get()));
                    return {std::move(res), std::move(h)};
                } 

            };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_SETUP_HPP */

