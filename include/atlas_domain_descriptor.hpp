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

// Check includes!

#include <vector>
#include <cassert>

#include "atlas/field/Field.h"

#include "./unstructured_grid.hpp"

namespace gridtools {

    // forward declaration
    template<typename P, typename DomainId>
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
            const atlas::Field& partition() { return m_partition; }
            const atlas::Field& remote_index() { return m_remote_index; }
            atlas::idx_t size() const { return m_size; }

    };

    /** @brief halo generator for atlas domains
     * An Atlas domain has already the notion of halos.
     * The purpose of the Atlas halo generator is to fill a container
     * with indexes referring to remote partitions.
     * @tparam P transport protocol
     * @tparam DomainId domain id type*/
    template<typename P, typename DomainId>
    class atlas_halo_generator {

        public:

            // member types
            using communicator_type = typename P::communicator;
            using domain_type = atlas_domain_descriptor<DomainId>;

        private:

            //members
            communicator_type m_comm;

        public:

            // ctors
            /** @brief construct a halo generator*/
            atlas_halo_generator(const communicator_type& comm) :
                m_comm{comm} {}

            // member functions
            /** @brief generate halos
             * @param d local domain instance
             * @return */
            auto operator()(const domain_type& domain) const {

                std::vector<atlas::idx_t> halo{};
                const auto rank = m_comm.rank();

                for (atlas::idx_t idx = 0; idx < domain.size(); ++idx) {
                    if (domain.partition()[idx] != rank) halo.push_back(idx);
                }

                return halo;

            }

    };

} // namespac gridtools

#endif /* INCLUDED_STRUCTURED_DOMAIN_DESCRIPTOR_HPP */
