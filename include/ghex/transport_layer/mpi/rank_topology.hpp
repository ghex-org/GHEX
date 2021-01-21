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
#include <iostream>
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
                    MPI_Comm m_local_comm;
                    MPI_Comm m_gather_comm;

                    int m_rank_global;
                    int m_size_global;
                    int m_rank_local;
                    int m_size_local;
                    int m_rank_gather;
                    int m_size_gather;
                    int m_rank_node;

                    std::vector<int> m_ranks_global;
                    std::set<int> m_ranks_global_set;

                    int m_master_rank_global;
                    int m_master_rank_local;
                    bool m_is_master;
                    int m_num_nodes;
                    
                    std::vector<int> m_sizes_local;
                    std::vector<int> m_sizes_local_offsets;

                    //set_type m_rank_set;
                    //int m_local_rank;
                    //int m_local_size;
                    //int m_local_master_rank;
                    //int m_master_rank;
                    //bool m_master;
                    //int m_num_nodes;

                public: // ctors
                    /** @brief construct from MPI communicator */
                    rank_topology(MPI_Comm comm) : m_comm(comm) {
                        // get rank from comm
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(comm,&m_rank_global));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(comm,&m_size_global));

                        // split comm into shared memory comms
                        const int key = m_rank_global;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_split_type(comm,
                            MPI_COMM_TYPE_SHARED, key, MPI_INFO_NULL, &m_local_comm));

                        // get rank within shared memory comm and its size
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_local_comm,&m_rank_local));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_local_comm,&m_size_local));

                        // gather rank info from all ranks within shared comm
                        m_ranks_global.resize(m_size_local);
                        GHEX_CHECK_MPI_RESULT(MPI_Allgather(&m_rank_global, 1,
                            MPI_INT, m_ranks_global.data(), 1, MPI_INT, m_local_comm));

                        // insert into set
                        for (auto r : m_ranks_global)
                            m_ranks_global_set.insert(r);

                        // set master rank
                        m_master_rank_global = *(m_ranks_global_set.begin());
                        m_is_master = (m_master_rank_global == m_rank_global);
                        if (m_is_master)
                            m_master_rank_local = m_rank_local;
                        else
                            m_master_rank_local = std::find(m_ranks_global.begin(),
                                m_ranks_global.end(), m_master_rank_global) - m_ranks_global.begin();

                        // split comm into gather comm
                        const int color = (m_is_master ? 0 : 1);
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_split(comm, color, key, &m_gather_comm));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_gather_comm,&m_rank_gather));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_gather_comm,&m_size_gather));

                        // broadcast the number of nodes
                        if (m_is_master)
                        {
                            m_num_nodes = m_size_gather;
                            m_rank_node = m_rank_gather;
                        }
                        if (m_size_local > 1 )
                        {
                            GHEX_CHECK_MPI_RESULT(MPI_Bcast(&m_num_nodes, 1, MPI_INT,
                                m_master_rank_local, m_local_comm));
                            GHEX_CHECK_MPI_RESULT(MPI_Bcast(&m_rank_node, 1, MPI_INT,
                                m_master_rank_local, m_local_comm));
                        }

                        //std::cout
                        //    << "rank:   " << m_rank_global << " " << m_rank_local << " " << m_rank_gather
                        //    << " " << m_is_master << "\n"
                        //    << "master: " << m_master_rank_global << " " << m_master_rank_local << "\n"
                        //    << "num nodes = " << m_num_nodes << std::endl;


                        // allgather local size
                        m_sizes_local.resize(m_num_nodes);
                        if (m_is_master)
                        {
                            GHEX_CHECK_MPI_RESULT(MPI_Allgather(&m_size_local, 1, MPI_INT,
                                m_sizes_local.data(), 1, MPI_INT, m_gather_comm));
                        }
                        GHEX_CHECK_MPI_RESULT(MPI_Bcast(m_sizes_local.data(), m_num_nodes, MPI_INT,
                            m_master_rank_local, m_local_comm));
                        m_sizes_local_offsets.resize(m_num_nodes);
                        std::partial_sum(m_sizes_local.begin(), m_sizes_local.end(),
                            m_sizes_local_offsets.begin());
                    }

                    ~rank_topology()
                    {
                        MPI_Comm_free(&m_local_comm);
                        MPI_Comm_free(&m_gather_comm);
                    }

                    rank_topology(const rank_topology&) = default;
                    rank_topology(rank_topology&&) noexcept = default;
                    rank_topology& operator=(const rank_topology&) = default;
                    rank_topology& operator=(rank_topology&&) noexcept = default;

                public: // member functions
                    /** @brief return raw mpi communicator */
                    auto mpi_comm() const noexcept { return m_comm; }

                    /** @brief return local rank number */
                    int local_rank() const noexcept { return m_rank_local; }

                    /** @brief return number of ranks on this node */
                    int local_size() const noexcept { return m_size_local; }

                    /** @brief return whether rank is located on this node */
                    bool is_local(int rank) const noexcept { return m_ranks_global_set.find(rank) != m_ranks_global_set.end(); }

                    /** @brief return ranks on this node */
                    const set_type& local_ranks() const noexcept { return m_ranks_global_set; }

                    /** @brief return raw shared mpi communicator */
                    auto mpi_shared_comm() const noexcept { return m_local_comm; }

                    auto num_nodes() const noexcept { return m_num_nodes; }
                    auto is_master() const noexcept { return m_is_master; }
                    auto local_master_rank() const noexcept { return m_master_rank_local; }
                    auto mpi_gather_comm() const noexcept { return m_gather_comm; }
                    const auto& node_sizes() const noexcept { return m_sizes_local; }
                    const auto& node_offsets() const noexcept { return m_sizes_local_offsets; }
                    auto node_rank() const noexcept { return m_rank_node; }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} //namespace gridtools

#endif // INCLUDED_GHEX_TL_MPI_RANK_TOPOLOGY_HPP
