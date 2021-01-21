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

#include <vector>
#include <cassert>
#include <algorithm>
#include <iostream>
#include <numeric>

#include "./rank_topology.hpp"
#include "./communicator_base.hpp"
#include "./status.hpp"
#include "./future.hpp"

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

                struct all_gather_sizes_t
                {
                    int m_this_size;
                    int m_node_size;
                    int m_all_size;
                    std::vector<int> m_local_sizes;
                    std::vector<int> m_local_displs;
                    std::vector<int> m_all_sizes;
                    std::vector<int> m_all_displs;
                    std::vector<int> m_node_sizes;
                    std::vector<int> m_node_displs;

                    int size() const noexcept { return m_all_size; }
                    auto begin() noexcept { return m_all_sizes.begin(); }
                    auto begin() const noexcept { return m_all_sizes.cbegin(); }
                    auto end() noexcept { return m_all_sizes.end(); }
                    auto end() const noexcept { return m_all_sizes.cend(); }
                    auto& operator[](unsigned int i) noexcept { return m_all_sizes[i]; }
                    const auto& operator[](unsigned int i) const noexcept { return m_all_sizes[i]; }
                };

                all_gather_sizes_t all_gather_sizes(int size)
                {
                    all_gather_sizes_t s;
                    s.m_all_sizes = all_gather(size);
                    s.m_all_displs.resize(s.m_all_sizes.size(),0);
                    std::partial_sum(s.m_all_sizes.begin(),s.m_all_sizes.end()-1,
                        s.m_all_displs.begin()+1);
                    s.m_all_size = s.m_all_displs.back()+s.m_all_sizes.back();
                    s.m_node_sizes.reserve(m_rank_topology.num_nodes());
                    s.m_local_sizes.reserve(m_rank_topology.local_size());
                    s.m_local_displs.resize(m_rank_topology.local_size(),0);
                    int last_offset = 0;
                    int n = 0;
                    for (auto offset : m_rank_topology.node_offsets())
                    {
                        if (n==m_rank_topology.node_rank())
                        {
                            s.m_local_sizes.push_back(s.m_all_sizes[last_offset]);
                            for (int i=last_offset+1; i<offset; ++i)
                            {
                                s.m_local_sizes.push_back(s.m_all_sizes[i]);
                                s.m_local_displs[i-last_offset] = s.m_local_displs[i-last_offset-1]
                                    + s.m_local_sizes[i-last_offset-1]; 
                            }
                        }
                        s.m_node_sizes.push_back(s.m_all_sizes[last_offset]);
                        for (int i=last_offset+1; i<offset; ++i)
                            s.m_node_sizes.back() += s.m_all_sizes[i];
                        last_offset = offset;
                        ++n;
                    }
                    s.m_node_displs.resize(m_rank_topology.num_nodes(),0);
                    std::partial_sum(s.m_node_sizes.begin(),s.m_node_sizes.end()-1,
                        s.m_node_displs.begin()+1);
                    s.m_this_size = size;
                    s.m_node_size = s.m_node_sizes[m_rank_topology.node_rank()];
                    return s;
                }

                template<typename T>
                struct all_gather_skeleton_t
                {
                    std::vector<T> m_elements;
                    std::vector<T> m_node_elements;
                    std::vector<int> m_local_sizes;
                    std::vector<int> m_local_displs;
                    std::vector<int> m_node_sizes;
                    std::vector<int> m_node_displs;
                };

                template<typename T>
                all_gather_skeleton_t<T> all_gather_skeleton(const all_gather_sizes_t& sizes)
                {
                    all_gather_skeleton_t<T> s{
                        std::vector<T>(sizes.m_all_size),
                        std::vector<T>(sizes.m_node_size),
                        sizes.m_local_sizes,
                        sizes.m_local_displs,
                        sizes.m_node_sizes,
                        sizes.m_node_displs};
                    for (auto& x : s.m_local_sizes) x*= sizeof(T);
                    for (auto& x : s.m_local_displs) x*= sizeof(T);
                    for (auto& x : s.m_node_sizes) x*= sizeof(T);
                    for (auto& x : s.m_node_displs) x*= sizeof(T);
                    return s;
                }
                
                //template<typename T>
                //void all_gather(const std::vector<T>& payload,
                //    const all_gather_skeleton_t<T>& skeleton, std::vector<std::vector<T>>& result)
                //{

                //}

                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload,
                    const all_gather_sizes_t& sizes)
                {
                    auto skeleton = all_gather_skeleton<T>(sizes);
                    return all_gather(payload, skeleton, sizes);
                }

                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload,
                    all_gather_skeleton_t<T>& skeleton, const all_gather_sizes_t& sizes)
                {
                    std::vector<std::vector<T>> res(size());

                    // gather to node master
                    GHEX_CHECK_MPI_RESULT(MPI_Gatherv(
                        payload.data(), payload.size()*sizeof(T), MPI_BYTE,
                        skeleton.m_node_elements.data(), 
                        skeleton.m_local_sizes.data(),
                        skeleton.m_local_displs.data(),
                        MPI_BYTE,
                        m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm()));

                    // all gather among masters
                    if (m_rank_topology.is_master())
                    {
                        GHEX_CHECK_MPI_RESULT(MPI_Allgatherv(        
                            skeleton.m_node_elements.data(), skeleton.m_node_elements.size()*sizeof(T),
                            MPI_BYTE, skeleton.m_elements.data(), skeleton.m_node_sizes.data(),
                            skeleton.m_node_displs.data(), MPI_BYTE, m_rank_topology.mpi_gather_comm()));
                    }

                    // broadcast
                    GHEX_CHECK_MPI_RESULT(MPI_Bcast(skeleton.m_elements.data(),
                        skeleton.m_elements.size()*sizeof(T), MPI_BYTE,
                        m_rank_topology.local_master_rank(), m_rank_topology.mpi_shared_comm()));

                    // reshuffle data
                    for (int i = 0; i<size(); ++i)
                    {
                        res[i].insert(res[i].end(),
                        skeleton.m_elements.begin()+sizes.m_all_displs[i],
                        skeleton.m_elements.begin()+sizes.m_all_displs[i]+sizes.m_all_sizes[i]);
                    }
                    return res;
                }

                template<typename T>
                std::vector<std::vector<T>> all_gather(const std::vector<T>& payload, const std::vector<int>& sizes) const
                {
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

