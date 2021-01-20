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
#ifndef INCLUDED_GHEX_TL_MPI_RANK_TOPOLOGY_HPP
#define INCLUDED_GHEX_TL_MPI_RANK_TOPOLOGY_HPP

#include <vector>
#include <set>
#include "./error.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace mpi {
                
                /** @brief Class representing node (shared memory) topology. */
                class rank_topology {
                public: // member types
                    using set_type  = std::set<int>;

                private: // members
                    MPI_Comm m_comm;
                    MPI_Comm m_shared_comm;
                    MPI_Comm m_master_comm;
                    set_type m_rank_set;
                    int m_local_rank;
                    int m_local_size;
                    int m_local_master_rank;
                    int m_master_rank;
                    bool m_master;
                    int m_num_nodes;

                public: // ctors
                    /** @brief construct from MPI communicator */
                    rank_topology(MPI_Comm comm) : m_comm(comm) {
                        // get rank from comm
                        int rank;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(comm,&rank));

                        // split comm into shared memory comms
                        const int key = rank;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_split_type(comm, MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &m_shared_comm));

                        // get rank within shared memory comm and its size
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_shared_comm,&m_local_rank));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_shared_comm,&m_local_size));

                        // gather rank info from all ranks within shared comm
                        std::vector<int> ranks(m_local_size);
                        GHEX_CHECK_MPI_RESULT(MPI_Allgather(
                            &rank, 1, MPI_INT, ranks.data(), 1, MPI_INT, m_shared_comm));

                        // insert into set
                        for (auto r : ranks) m_rank_set.insert(r);

                        m_master_rank = *m_rank_set.begin();
                        m_master = (m_master_rank == rank);

                        // split comm into gather comm
                        const int color = (m_master ? 0 : 1);
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_split_type(comm, color, key, MPI_INFO_NULL, &m_master_comm));

                        // get local master rank and count the number of nodes
                        if (m_master)
                        {
                            m_local_master_rank = m_local_rank;
                            GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_master_comm,&m_num_nodes));
                        }

                        // broadcast to the other ranks within the node
                        if (m_local_size>1)
                        {
                            GHEX_CHECK_MPI_RESULT(MPI_Bcast(&m_local_master_rank, 1, MPI_INT, 0, m_shared_comm));
                            GHEX_CHECK_MPI_RESULT(MPI_Bcast(&m_num_nodes, 1, MPI_INT, 0, m_shared_comm));
                        }

                        // allgather local size
                        if (m_master)
                        {
                            std::vector<int> local_sizes(m_num_nodes);
                            GHEX_CHECK_MPI_RESULT(MPI_Allgather(
                                &m_local_size, 1, MPI_INT, local_sizes.data(), 1, MPI_INT, m_master_comm));

                            std::vector<int> recvbuf;
                            std::vector<int> displs(m_num_nodes);
                            std::vector<std::vector<int>> res(m_num_nodes);
                            unsigned int c = 0;
                            for (int i=0; i<m_num_nodes; ++i)
                            {
                                displs[i] = c;
                                res[i].resize(local_sizes[i]);
                                c += local_sizes[i];
                            }
                            recvbuf.resize(c);

                            GHEX_CHECK_MPI_RESULT(MPI_Allgatherv(
                                &ranks[0], ranks.size(), MPI_INT,
                                &recvbuf[0], &local_sizes[0], &displs[0], MPI_INT,
                                m_master_comm));
                        }
                    }

                    ~rank_topology()
                    {
                        MPI_Comm_free(&m_shared_comm);
                        MPI_Comm_free(&m_master_comm);
                    }

                    rank_topology(const rank_topology&) = default;
                    rank_topology(rank_topology&&) noexcept = default;
                    rank_topology& operator=(const rank_topology&) = default;
                    rank_topology& operator=(rank_topology&&) noexcept = default;

                public: // member functions
                    /** @brief return raw mpi communicator */
                    auto mpi_comm() const noexcept { return m_comm; }

                    /** @brief return local rank number */
                    int local_rank() const noexcept { return m_local_rank; }

                    /** @brief return number of ranks on this node */
                    int local_size() const noexcept { return m_local_size; }

                    /** @brief return whether rank is located on this node */
                    bool is_local(int rank) const noexcept { return m_rank_set.find(rank) != m_rank_set.end(); }

                    int local_master_rank() const noexcept { return m_local_master_rank; }

                    auto num_nodes() const noexcept { return m_num_nodes; }

                    /** @brief return ranks on this node */
                    const set_type& local_ranks() const noexcept { return m_rank_set; }

                    /** @brief return raw shared mpi communicator */
                    auto mpi_shared_comm() const noexcept { return m_shared_comm; }

                    bool is_node_master() const noexcept { return m_master; }

                    /** @brief return raw gather mpi communicator */
                    auto mpi_master_comm() const noexcept { return m_master_comm; }
                    
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} //namespace gridtools

#endif // INCLUDED_GHEX_TL_MPI_RANK_TOPOLOGY_HPP
