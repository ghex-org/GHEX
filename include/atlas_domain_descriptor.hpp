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
     * @tparam DomainId domain id type*/
    template<typename DomainId>
    class atlas_domain_descriptor {

        public:

            // member types
            using domain_id_type = DomainId;

        private:

            // members
            domain_id_type m_id;
            atlas::Field m_partition;
            atlas::Field m_remote_index;
            atlas::idx_t m_size;

        public:

            // ctors
            // NOTE: a constructor which takes as argument a reference to the whole mesh
            // may not be a good idea, since it has to be spoecialized for a given function space.
            /** @brief construct a local domain
             * @param id domain id
             * @param partition field with partition indexes
             * @param remote_index field with local indexes in remote partition*/
            atlas_domain_descriptor(domain_id_type id,
                                    const atlas::Field& partition,
                                    const atlas::Field& remote_index,
                                    const atlas::idx_t size) :
                m_id{id},
                m_partition{partition},
                m_remote_index{remote_index},
                m_size{size} {
                assert(size == m_partition.size());
                assert(size == remote_index.size());
            }

            // member functions
            domain_id_type domain_id() const { return m_id; }
            const atlas::Field& partition() const { return m_partition; }
            const atlas::Field& remote_index() const { return m_remote_index; }
            atlas::idx_t size() const { return m_size; }

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

        private:

            //members
            int m_rank;

        public:

            // ctors
            /** @brief construct a halo generator*/
            atlas_halo_generator(const int rank) :
                m_rank{rank} {}

            // member functions
            /** @brief generate halos
             * @param domain local domain instance
             * @return halo (vector of halo indexes)*/
            auto operator()(const domain_type& domain) const {

                std::vector<atlas::idx_t> halo{};
                const int* partition_data = atlas::array::make_view<int, 1>(domain.partition()).data();

                for (auto idx = 0; idx < domain.size(); ++idx) {
                    if (partition_data[idx] != m_rank) halo.push_back(idx);
                }

                return halo;

            }

    };

} // namespac gridtools

#endif /* INCLUDED_STRUCTURED_DOMAIN_DESCRIPTOR_HPP */
