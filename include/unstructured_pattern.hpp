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
#ifndef INCLUDED_UNSTRUCTURED_PATTERN_HPP
#define INCLUDED_UNSTRUCTURED_PATTERN_HPP

#include "./protocol/communicator_base.hpp"
#include "./pattern.hpp"
#include "./unstructured_grid.hpp"

namespace gridtools {

    /** @brief unstructured pattern specialization
     *
     * This class provides access to the receive and send iteration spaces, determined by the halos,
     * and holds all connections to the neighbors.
     *
     * @tparam P transport protocol
     * @tparam Index index type for domain and iteration space
     * @tparam DomainId domain id type*/
    template<typename P, typename Index, typename DomainId>
    class pattern<P, detail::unstructured_grid<Index>, DomainId> {

        public:

            // member types

            using communicator_type = protocol::communicator<P>;
            using address_type = typename communicator_type::address_type;
            using grid_type = detail::unstructured_grid<Index>;
            using domain_id_type = DomainId;
            using pattern_container_type = pattern_container<P, grid_type, domain_id_type>;

            friend class pattern_container<P, grid_type, domain_id_type>;

            struct iteration_space {

                // members

                // ctors
                iteration_space() noexcept = default;
                iteration_space(const iteration_space&) = default;
                iteration_space(iteration_space&&) = default;

                // operators
                iteration_space& operator = (const iteration_space& other) noexcept = default;
                iteration_space& operator = (iteration_space&& other) noexcept = default;

                // member functions

                /** @brief number of elements*/
                int size() const noexcept {
                    return 0;
                }

                /** @brief print */
                template<class CharT, class Traits>
                friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const iteration_space& is) {
                    // ...
                    return os;
                }

            };

            using iteration_space_pair = iteration_space;
            using index_container_type = std::vector<iteration_space_pair>;

            /** @brief extended domain id, including rank and tag information used as key in halo lookup map*/
            struct extended_domain_id_type {

                // members
                domain_id_type id;
                int mpi_rank;
                address_type address;
                int tag;

                // member functions

                /** @brief unique ordering given by id and tag*/
                bool operator < (const extended_domain_id_type& other) const noexcept {
                    return (id < other.id ? true : (id == other.id ? (tag < other.tag) : false));
                }

                /** @brief print*/
                template<class CharT, class Traits>
                    friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const extended_domain_id_type& dom_id) {
                        // ...
                        return os;
                    }

            };

            // halo map type
            using map_type = std::map<extended_domain_id_type, index_container_type>;

            // static member functions

            /** @brief compute number of elements in an object of type index_container_type */
            static int num_elements(const index_container_type& c) noexcept {
                std::size_t s{0};
                for (const auto& is : c) s += is.size();
                return s;
            }

        private:

            // members
            communicator_type m_comm;
            iteration_space_pair m_domain;
            extended_domain_id_type m_id;
            map_type m_send_map;
            map_type m_recv_map;
            pattern_container_type* m_container;

        public:

            // ctors
            pattern(communicator_type& comm, const iteration_space_pair& domain, const extended_domain_id_type& id) :
                m_comm(comm),
                m_domain(domain),
                m_id(id) {}
            pattern(const pattern&) = default;
            pattern(pattern&&) = default;

            // member functions
            communicator_type& communicator() noexcept { return m_comm; }
            const communicator_type& communicator() const noexcept { return m_comm; }
            domain_id_type domain_id() const noexcept { return m_id.id; }
            extended_domain_id_type extended_domain_id() const noexcept { return m_id; }
            map_type& send_halos() noexcept { return m_send_map; }
            const map_type& send_halos() const noexcept { return m_send_map; }
            map_type& recv_halos() noexcept { return m_recv_map; }
            const map_type& recv_halos() const noexcept { return m_recv_map; }
            const pattern_container_type& container() const noexcept { return *m_container; }

    };

    namespace detail {

        /** @brief construct pattern with the help of all to all communication*/
        template<typename Index>
        struct make_pattern_impl<detail::unstructured_grid<Index>> {

            template<typename P, typename HaloGenerator, typename DomainRange>
            static auto apply(protocol::setup_communicator& comm, protocol::communicator<P>& new_comm, HaloGenerator&& hgen, DomainRange&& d_range) {

                // typedefs
                using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
                using domain_id_type = typename domain_type::domain_id_type;
                using grid_type = detail::unstructured_grid<Index>;
                using pattern_type = pattern<P, grid_type, domain_id_type>;

                // get this address from new communicator
                auto my_address = new_comm.address();

                std::vector<pattern_type> my_patterns;

                // needed with multiple domains per PE
                int m_max_tag = 0;

                // ...

                return pattern_container<P, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

            }

        };

    } // namespace detail

} // namespace gridtools

#endif /* INCLUDED_UNSTRUCTURED_PATTERN_HPP */
