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
#ifndef INCLUDED_ATLAS_DOMAIN_DESCRIPTOR_HPP
#define INCLUDED_ATLAS_DOMAIN_DESCRIPTOR_HPP

#include <vector>
#include <cassert>

#include "atlas/field/Field.h"
#include "atlas/array.h"

#include "./unstructured_grid.hpp"

namespace gridtools {

    // forward declaration
    template<typename DomainId>
    class atlas_halo_generator;

    /** @brief implements domain descriptor concept for Atlas domains
     * An Atlas domain has inner notion of the halos.
     * It has to be istantiated using a mesh which has already grown the required halo layer
     * due to the creation of a function space with a halo.
     * Null halo is fine too, provided that the mesh is in its final state.
     * Be careful: domain size includes halos!
     * @tparam DomainId domain id type*/
    template<typename DomainId>
    class atlas_domain_descriptor {

        public:

            // member types
            using domain_id_type = DomainId;
            using index_t = atlas::idx_t;

        private:

            // members
            domain_id_type m_id;
            atlas::Field m_partition;
            atlas::Field m_remote_index;
            index_t m_size;
            index_t m_first;
            index_t m_last;

        public:

            // ctors
            // NOTE: a constructor which takes as argument a reference to the whole mesh
            // may not be a good idea, since it has to be spoecialized for a given function space.
            /** @brief construct a local domain
             * @param id domain id
             * @param partition field with partition indexes
             * @param remote_index field with local indexes in remote partition
             * @param size size of the domain + all required halo points
             * @param rank PE rank*/
            atlas_domain_descriptor(domain_id_type id,
                                    const atlas::Field& partition,
                                    const atlas::Field& remote_index,
                                    const index_t size,
                                    const int rank) :
                m_id{id},
                m_partition{partition},
                m_remote_index{remote_index},
                m_size{size} {

                // Asserts
                assert(size > 0);
                assert(size == partition.size());
                assert(size == remote_index.size());

                // Setup first and last (with the assumption that first/last coincides with min/max)
                const int* partition_data = atlas::array::make_view<int, 1>(m_partition).data();
                const index_t* remote_index_data = atlas::array::make_view<index_t, 1>(m_remote_index).data();
                index_t first = remote_index_data[0];
                index_t last = remote_index_data[0];
                for (auto idx = 1; idx < m_size; ++idx) {
                    if (partition_data[idx] == rank) {
                        if (remote_index_data[idx] < first) {
                            first = remote_index_data[idx];
                        } else if (remote_index_data[idx] > last) {
                            last = remote_index_data[idx];
                        }
                    }
                }
                m_first = first;
                m_last = last;

            }

            // member functions
            domain_id_type domain_id() const noexcept { return m_id; }
            const atlas::Field& partition() const noexcept { return m_partition; }
            const atlas::Field& remote_index() const noexcept { return m_remote_index; }
            index_t size() const noexcept { return m_size; }
            /** @brief first local index, excluding halo points*/
            index_t first() const noexcept { return m_first; }
            /** @brief last local index, excluding halo points*/
            index_t last() const noexcept { return m_last; }

            // print
            /** @brief print */
            template<class CharT, class Traits>
            friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const atlas_domain_descriptor& domain) {
                os << "domain id = " << domain.domain_id() << ";\n"
                   << "size = " << domain.size() << ";\n"
                   << "partition indexes: [" << domain.partition() << "]\n"
                   << "remote indexes: [" << domain.remote_index() << "]\n";
                return os;
            }

    };

    /** @brief halo generator for atlas domains
     * An Atlas domain has already the notion of halos.
     * The purpose of the Atlas halo generator is to fill a container
     * with indexes referring to remote partitions.
     * @tparam DomainId domain id type*/
    template<typename DomainId>
    class atlas_halo_generator {

        public:

            // member types
            using domain_type = atlas_domain_descriptor<DomainId>;
            using index_t = atlas::idx_t;

            /** @brief essentially a partition index and a sequence of remote indexes;
             * conceptually, it is similar to an iteration space
             * (only differences are that here it is already specialized for atlas
             * and provides an insert method as well),
             * and this is why the two concepts could potentially coincide*/
            class halo {

                private:

                    int m_partition;
                    std::vector<index_t> m_remote_index;

                public:

                    // ctors
                    halo() noexcept = default;
                    halo(const int partition) noexcept :
                        m_partition{partition},
                        m_remote_index{} {}
                    halo(const int partition, const std::vector<index_t>& remote_index) noexcept :
                        m_partition{partition},
                        m_remote_index{remote_index} {}

                    // member functions
                    int partition() const noexcept { return m_partition; }
                    std::vector<index_t>& remote_index() noexcept { return m_remote_index; }
                    const std::vector<index_t>& remote_index() const noexcept { return m_remote_index; }
                    std::size_t size() const noexcept { return m_remote_index.size(); }
                    void push_back(const index_t remote_index_idx) {
                        m_remote_index.push_back(remote_index_idx);
                    }

                    // print
                    /** @brief print */
                    template<class CharT, class Traits>
                    friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const halo& h) {
                        os << "size = " << h.size() << ";\n"
                           << "partition = " << h.partition() << ";\n"
                           << "remote indexes: [ ";
                        for (auto idx : h.remote_index()) { os << idx << " "; }
                        os << "]\n";
                        return os;
                    }

            };

        private:

            //members
            int m_rank;
            int m_size;

        public:

            // ctors
            /** @brief construct a halo generator
             * @param rank PE rank
             * @param size number of PEs*/
            atlas_halo_generator(const int rank, const int size) noexcept :
                m_rank{rank},
                m_size{size} {}

            // member functions
            /** @brief generate halos
             * @param domain local domain instance
             * @return vector of receive halos, one for each PE*/
            auto operator()(const domain_type& domain) const {

                std::vector<halo> halos{};
                for (int rank = 0; rank < m_size; ++rank) {
                    halos.push_back({rank});
                }

                const int* partition_data = atlas::array::make_view<int, 1>(domain.partition()).data();
                const index_t* remote_index_data = atlas::array::make_view<index_t, 1>(domain.remote_index()).data();

                // if the index refers to another rank, or even to the same rank but as a halo point,
                // corresponding halo is updated
                for (auto domain_idx = 0; domain_idx < domain.size(); ++domain_idx) {
                    if ((partition_data[domain_idx] != m_rank) || (remote_index_data[domain_idx] != domain_idx)) {
                        halos[partition_data[domain_idx]].push_back(remote_index_data[domain_idx]);
                    }
                }

                return halos;

            }

    };

} // namespac gridtools

#endif /* INCLUDED_STRUCTURED_DOMAIN_DESCRIPTOR_HPP */
