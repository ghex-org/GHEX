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
#include <unordered_set>
#include "./error.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace mpi {
                
                /** @brief Class representing node (shared memory) topology. */
                class rank_topology {
                public: // member types
                    using set_type  = std::unordered_set<int>;
                    using size_type = set_type::size_type;

                private: // members
                    MPI_Comm m_comm;
                    MPI_Comm m_shared_comm;
                    int m_rank;
                    std::unordered_set<int> m_rank_set;

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
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(m_shared_comm,&m_rank));
                        int size;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_size(m_shared_comm,&size));
                        // gather rank info from all ranks within shared comm
                        std::vector<int> ranks(size);
                        MPI_Allgather(&rank, 1, MPI_INT, ranks.data(), 1, MPI_INT, m_shared_comm);
                        // insert into set
                        for (auto r : ranks) m_rank_set.insert(r);
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_free(&m_shared_comm));
                    }

                    rank_topology(const rank_topology&) = default;
                    rank_topology(rank_topology&&) noexcept = default;
                    rank_topology& operator=(const rank_topology&) = default;
                    rank_topology& operator=(rank_topology&&) noexcept = default;

                public: // member functions
                    /** @brief return whether rank is located on this node */
                    bool is_local(int rank) const noexcept { return m_rank_set.find(rank) != m_rank_set.end(); }

                    /** @brief return number of ranks on this node */
                    size_type local_size() const noexcept { return m_rank_set.size(); }

                    /** @brief return ranks on this node */
                    const set_type& local_ranks() const noexcept { return m_rank_set; }

                    /** @brief return local rank number */
                    int local_rank() const noexcept { return m_rank; }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} //namespace gridtools

#endif // INCLUDED_GHEX_TL_MPI_RANK_TOPOLOGY_HPP
