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
#ifndef INCLUDED_GHEX_TL_MPI_SETUP_HPP
#define INCLUDED_GHEX_TL_MPI_SETUP_HPP

#include "./rank_topology.hpp"
#include "./communicator_base.hpp"
#include "./status.hpp"
#include "./future.hpp"
#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

            /** @brief special mpi communicator used for setup phase */
            class setup_communicator
            : public communicator_base
            {
            public:
                using base_type    = communicator_base;
                using handle_type  = request_t;
                using address_type = base_type::rank_type;
                using status       = status_t;
                template<typename T>
                using future = future_t<T>;

            private:
                const rank_topology& m_rank_topology;

            public:
                setup_communicator(const rank_topology& t)
                : base_type{t.mpi_comm()}
                , m_rank_topology(t) 
                {}
                setup_communicator(const setup_communicator&) = default;
                //setup_communicator& operator=(const setup_communicator&) = default;
                setup_communicator(setup_communicator&&) noexcept = default;
                //setup_communicator& operator=(setup_communicator&&) noexcept = default;

                address_type address() const { return rank(); }

                template<typename T>
                void send(int dest, int tag, const T & value) const
                {
                    GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(&value), sizeof(T), MPI_BYTE, dest, tag, *this));
                }

                template<typename T>
                status recv(int source, int tag, T & value) const
                {
                    MPI_Status status;
                    GHEX_CHECK_MPI_RESULT(MPI_Recv(reinterpret_cast<void*>(&value), sizeof(T), MPI_BYTE, source, tag, *this, &status));
                    return {status};
                }

                template<typename T>
                void send(int dest, int tag, const T* values, int n) const
                {
                    GHEX_CHECK_MPI_RESULT(MPI_Send(reinterpret_cast<const void*>(values), sizeof(T)*n, MPI_BYTE, dest, tag, *this));
                }

                template<typename T>
                status recv(int source, int tag, T* values, int n) const
                {
                    MPI_Status status;
                    GHEX_CHECK_MPI_RESULT(MPI_Recv(reinterpret_cast<void*>(values), sizeof(T)*n, MPI_BYTE, source, tag, *this, &status));
                    return {status};
                }

                template<typename T> 
                void broadcast(T& value, int root) const
                {
                    GHEX_CHECK_MPI_RESULT(MPI_Bcast(&value, sizeof(T), MPI_BYTE, root, *this));
                }

                template<typename T> 
                void broadcast(T * values, int n, int root) const
                {
                    GHEX_CHECK_MPI_RESULT(MPI_Bcast(values, sizeof(T)*n, MPI_BYTE, root, *this));
                }

#ifdef GHEX_EMULATE_ALLGATHERV
                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload, const std::vector<int>& sizes) const
                {
                    std::vector<std::vector<T>> res(size());
                    for (int neigh=0; neigh<size(); ++neigh)
                    {
                        res[neigh].resize(sizes[neigh]);
                    }
                    std::vector<handle_type> m_reqs;
                    m_reqs.reserve((size())*2);
                    for (int neigh=0; neigh<size(); ++neigh)
                    {
                        m_reqs.push_back( handle_type{} );
                        GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(res[neigh].data()), sizeof(T)*sizes[neigh], MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
                    }
                    for (int neigh=0; neigh<size(); ++neigh)
                    {
                        m_reqs.push_back( handle_type{} );
                        GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(payload.data()), sizeof(T)*payload.size(), MPI_BYTE, neigh, 99, *this, &m_reqs.back().get()));
                    }
                    for (auto& r : m_reqs)
                        r.wait();

                    return res;
                }
#else
                struct all_gather_sizes_t
                {
                    bool m_master;
                    int m_this_size;
                    std::vector<int> m_local_sizes;
                    int m_node_size;
                    std::vector<int> m_node_sizes;
                };

                all_gather_sizes_t all_gather_sizes(int size)
                {
                    all_gather_sizes_t s;
                    if (m_rank_topology.is_node_master())
                        s.m_master = true;
                    else
                        s.m_master = false;
                    s.m_this_size = size;
                    if (m_rank_topology.local_size() > 1)
                    {
                        if (m_rank_topology.is_node_master())
                            s.m_local_sizes.resize(m_rank_topology.local_size());
                        GHEX_CHECK_MPI_RESULT(
                            MPI_Gather(&size, 1, MPI_INT,
                            &s.m_local_sizes[0], 1, MPI_INT,
                            m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm()));
                    }
                    else
                    {
                        s.m_local_sizes.push_back(size);
                    }
                    if (m_rank_topology.is_node_master())
                    {
                        s.m_node_size = std::accumulate(s.m_local_sizes.begin(), s.m_local_sizes.end(),0);
                        s.m_node_sizes.resize(m_rank_topology.num_nodes());
                        GHEX_CHECK_MPI_RESULT(
                            MPI_Allgather(&s.m_node_size, 1, MPI_INT,
                            &s.m_node_sizes[0], 1, MPI_INT, m_rank_topology.mpi_master_comm()));
                    }
                    return s;
                }

                template<typename T>
                struct all_gather_skeleton_t
                {
                    std::vector<T> m_node_elements;
                    std::vector<T> m_all_elements;
                };

                template<typename T>
                all_gather_skeleton_t all_gather_skeleton(const all_gather_sizes_t& sizes)
                {

                }
                
                template<typename T>
                void all_gather(const std::vector<T>& payload,
                    const all_gather_skeleton_t<T>& skeleton, std::vector<std::vector<T>>& result)
                {

                }

                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload,
                    const all_gather_skeleton_t<T>& skeleton)
                {

                }

                //template<typename T>
                //std::vector<std::vector<T>> local_gather(const std::vector<T>& payload, const std::vector<int>& sizes) const
                //{
                //    if (m_rank_topology.is_node_master())
                //    {

                //    }
                //    GHEX_CHECK_MPI_RESULT(MPI_Gatherv(
                //        payload.data(), payload.size()*sizeof(T), MPI_BYTE,
                //        void *recvbuf, const int *recvcounts, const int *displs, MPI_BYTE,
                //        m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm()));

                //}

                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload, const std::vector<int>& sizes) const
                {
                    if (m_rank_topology.local_size() > 1)
                    {
                        if (m_rank_topology.is_node_master())
                        {
                            std::cout << "do gather within rank first" << std::endl;
                            std::vector<int> local_sizes;
                            local_sizes.reserve(m_rank_topology.local_size());
                            for (auto r : m_rank_topology.local_ranks())
                                local_sizes.push_back(sizes[r]);


                            //GHEX_CHECK_MPI_RESULT(
                            //    MPI_Gatherv(
                            //        payload.data(), payload.size()*sizeof(T), MPI_BYTE,
                            //        void *recvbuf, const int *recvcounts, const int *displs,
                            //        m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm())
                            //);
                        }
                        else
                        {
                            GHEX_CHECK_MPI_RESULT(
                                MPI_Gatherv(
                                    payload.data(), payload.size()*sizeof(T), MPI_BYTE,
                                    (void*)0, 0, (const int *)0, MPI_BYTE,
                                    m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm())
                            );
                        }
                    }
                    std::vector<T> recvbuf;
                    std::vector<int> recvcounts(size());
                    std::vector<int> displs(size());
                    std::vector<std::vector<T>> res(size());
                    unsigned int c = 0;
                    for (int i=0; i<size(); ++i)
                    {
                        recvcounts[i] = sizes[i]*sizeof(T);
                        displs[i] = c*sizeof(T);
                        res[i].resize(sizes[i]);
                        c += sizes[i];
                    }
                    recvbuf.resize(c);

                    GHEX_CHECK_MPI_RESULT( MPI_Allgatherv(
                        &payload[0], payload.size()*sizeof(T), MPI_BYTE,
                        &recvbuf[0], &recvcounts[0], &displs[0], MPI_BYTE,
                        *this));

                    c = 0;
                    for (int i=0; i<size(); ++i)
                    {
                        std::copy(
                            &recvbuf[displs[i]/sizeof(T)],
                            &recvbuf[displs[i]/sizeof(T)] + sizes[i],
                            res[i].begin());
                    }
                    return res;
                }
#endif

                template<typename T>
                std::vector<T> all_gather(const T& payload) const
                {
                    std::vector<T> res(size());
                    GHEX_CHECK_MPI_RESULT(
                        MPI_Allgather
                        (&payload, sizeof(T), MPI_BYTE,
                        &res[0], sizeof(T), MPI_BYTE,
                        *this));
                    return res;
                }
                
                /** @brief computes the max element of a vector<T> among all ranks */
                template<typename T>
                T max_element(const std::vector<T>& elems) const {
                    T local_max{*(std::max_element(elems.begin(), elems.end()))};
                    auto all_max = all_gather(local_max);
                    return *(std::max_element(all_max.begin(), all_max.end()));
                }

                /** @brief just a helper function using custom types to be used when send/recv counts can be deduced*/
                template<typename T>
                void all_to_all(const std::vector<T>& send_buf, std::vector<T>& recv_buf) const
                {
                    int comm_size = this->size();
                    assert(send_buf.size() % comm_size == 0);
                    assert(recv_buf.size() % comm_size == 0);
                    int send_count = send_buf.size() / comm_size * sizeof(T);
                    int recv_count = recv_buf.size() / comm_size * sizeof(T);
                    GHEX_CHECK_MPI_RESULT(
                            MPI_Alltoall
                            (reinterpret_cast<const void*>(send_buf.data()), send_count, MPI_BYTE,
                             reinterpret_cast<void*>(recv_buf.data()), recv_count, MPI_BYTE, *this));
                }
                
                /** @brief just a wrapper using custom types*/
                template<typename T>
                void all_to_allv(const std::vector<T>& send_buf, const std::vector<int>& send_counts, const std::vector<int>& send_displs,
                        std::vector<T>& recv_buf, const std::vector<int>& recv_counts, const std::vector<int>& recv_displs) const
                {
                    int comm_size = this->size();
                    std::vector<int> send_counts_b(comm_size), send_displs_b(comm_size), recv_counts_b(comm_size), recv_displs_b(comm_size);
                    for (auto i=0; i<comm_size; ++i) send_counts_b[i] = send_counts[i] * sizeof(T);
                    for (auto i=0; i<comm_size; ++i) send_displs_b[i] = send_displs[i] * sizeof(T);
                    for (auto i=0; i<comm_size; ++i) recv_counts_b[i] = recv_counts[i] * sizeof(T);
                    for (auto i=0; i<comm_size; ++i) recv_displs_b[i] = recv_displs[i] * sizeof(T);
                    GHEX_CHECK_MPI_RESULT(
                            MPI_Alltoallv
                            (reinterpret_cast<const void*>(send_buf.data()), &send_counts_b[0], &send_displs_b[0], MPI_BYTE,
                             reinterpret_cast<void*>(recv_buf.data()), &recv_counts_b[0], &recv_displs_b[0], MPI_BYTE,
                             *this));
                }
            
            };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_SETUP_HPP */

