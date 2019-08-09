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

#include <vector>
#include <cassert>
#include <numeric>

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
            using index_type = Index;
            using grid_type = detail::unstructured_grid<index_type>;
            using domain_id_type = DomainId;
            using pattern_container_type = pattern_container<P, grid_type, domain_id_type>;

            friend class pattern_container<P, grid_type, domain_id_type>;

            /** @brief essentially a partition index and a sequence of remote indexes
             * WARN: a second index, and therefore a more complex iteration space,
             * is needed to handle correctly multiple vertical layers*/
            class iteration_space {

                private:

                    int m_partition;
                    std::vector<index_type> m_remote_index;

                public:

                    // ctors
                    iteration_space() noexcept = default;
                    iteration_space(const int partition, const std::vector<index_type>& remote_index) noexcept :
                        m_partition{partition},
                        m_remote_index{remote_index} {}
                    iteration_space(const int partition, std::vector<index_type>&& remote_index) noexcept :
                        m_partition{partition},
                        m_remote_index{std::move(remote_index)} {}
                    // less safe but maybe preferable
                    // template<typename V>
                    // iteration_space(const int partition, V&& remote_index) noexcept :
                    //     m_partition{partition},
                    //     m_remote_index{std::forward<V>(remote_index)} {}
                    iteration_space(const int partition, const index_type first, const index_type last) noexcept :
                        m_partition{partition},
                        m_remote_index{} {
                        m_remote_index.resize(static_cast<std::size_t>(last - first + 1));
                        for (index_type idx = first; idx <= last; ++idx) {
                            m_remote_index[static_cast<std::size_t>(idx)] = idx;
                        }
                    }

                    // member functions
                    int partition() const noexcept { return m_partition; }
                    std::vector<index_type>& remote_index() noexcept { return m_remote_index; }
                    const std::vector<index_type>& remote_index() const noexcept { return m_remote_index; }
                    std::size_t size() const noexcept { return m_remote_index.size(); }

                    // print
                    /** @brief print */
                    template<class CharT, class Traits>
                    friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const iteration_space& is) {
                        os << "size = " << is.size() << ";\n"
                           << "partition = " << is.partition() << ";\n"
                           << "remote indexes: [ ";
                        for (auto idx : is.remote_index()) { os << idx << " "; }
                        os << "]\n";
                        return os;
                    }

            };

            using iteration_space_pair = iteration_space;
            using index_container_type = std::vector<iteration_space_pair>;

            /** @brief extended domain id, including rank and tag information
             * WARN: does not include actual domain id, differently from structured pattern case*/
            struct extended_domain_id_type {

                // members
                // domain_id_type id;
                int mpi_rank;
                address_type address;
                int tag;

                // member functions
                // /** @brief unique ordering given by id and tag*/
                // bool operator < (const extended_domain_id_type& other) const noexcept {
                //     return (id < other.id ? true : (id == other.id ? (tag < other.tag) : false));
                // }
                /** @brief unique ordering given by address and tag*/
                bool operator < (const extended_domain_id_type& other) const noexcept {
                    return (address < other.address ? true : (address == other.address ? (tag < other.tag) : false));
                }

                // print
                // /** @brief print*/
                // template<class CharT, class Traits>
                // friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const extended_domain_id_type& dom_id) {
                //     os << "{id=" << dom_id.id << ", tag=" << dom_id.tag << ", rank=" << dom_id.mpi_rank << "}";
                //     return os;
                // }
                /** @brief print*/
                template<class CharT, class Traits>
                friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const extended_domain_id_type& dom_id) {
                    os << "{tag=" << dom_id.tag << ", rank=" << dom_id.mpi_rank << "}";
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
                using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
                using index_container_type = typename pattern_type::index_container_type;
                using index_type = typename pattern_type::index_type;
                using address_type = typename pattern_type::address_type;

                // get this rank, address and size from new communicator
                auto my_rank = new_comm.rank(); // WARN: comm or new_comm?
                auto my_address = new_comm.address();
                size_t size = static_cast<std::size_t>(new_comm.size());

                std::vector<pattern_type> my_patterns;

                std::vector<int> recv_counts{};
                recv_counts.resize(size);
                std::vector<int> send_counts{};
                send_counts.resize(size);
                int recv_count;
                int send_count;
                std::vector<int> recv_displs{};
                recv_displs.resize(size);
                std::vector<int> send_displs{};
                send_displs.resize(size);
                std::vector<index_type> recv_indexes{};
                std::vector<index_type> send_indexes{};

                // needed with multiple domains per PE
                int m_max_tag = 0;

                for (const auto& d : d_range) { // WARN: so far, multiple domains are not fully supported

                    // setup pattern
                    pattern_type p{new_comm, {my_rank, d.first(), d.last()}, {my_rank, my_address, 0}};

                    std::fill(recv_counts.begin(), recv_counts.end(), 0);
                    std::fill(send_counts.begin(), send_counts.end(), 0);
                    std::fill(recv_displs.begin(), recv_displs.end(), 0);
                    std::fill(send_displs.begin(), send_displs.end(), 0);

                    // set up receive halos
                    auto generated_recv_halos = hgen(d);
                    for (const auto& h : generated_recv_halos) {
                        if (h.size()) {
                            // WARN: very simplified definition of extended domain id;
                            // a more complex one is needed for multiple domains
                            int tag = (h.partition() << 5) + my_address; // WARN: maximum address = 2^5 - 1
                            extended_domain_id_type id{h.partition(), static_cast<address_type>(h.partition()), tag}; // WARN: address is not obtained from the other domain
                            index_container_type ic{ {h.partition(), h.remote_index()} };
                            p.recv_halos().insert(std::make_pair(id, ic));
                            recv_counts[static_cast<std::size_t>(h.partition())] = static_cast<int>(h.size());
                        }
                    }

                    // set up all-to-all communication, receive side
                    recv_count = std::accumulate(recv_counts.begin(), recv_counts.end(), 0);
                    recv_indexes.resize(recv_count);
                    recv_displs[0] = 0;
                    for (std::size_t rank = 1; rank < size; ++rank) {
                        recv_displs[rank] = recv_displs[rank - 1] + recv_counts[rank - 1];
                    }
                    for (const auto& h : generated_recv_halos) {
                        if (h.size()) {
                            std::memcpy(&recv_indexes[static_cast<std::size_t>(recv_displs[static_cast<std::size_t>(h.partition())])],
                                    &h.remote_index()[0],
                                    h.size() * sizeof(index_type));
                        }
                    }

                    // set up all-to-all communication, send side
                    comm.allToAll(recv_counts, send_counts);
                    send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
                    send_indexes.resize(send_count);
                    send_displs[0] = 0;
                    for (std::size_t rank = 1; rank < size; ++rank) {
                        send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
                    }

                    // set up send halos
                    comm.allToAllv(&recv_indexes[0], &recv_counts[0], &recv_displs[0],
                            &send_indexes[0], &send_counts[0], &send_displs[0]);
                    for (std::size_t rank = 0; rank < size; ++rank) {
                        if (send_counts[rank]) {
                            // WARN: very simplified definition of extended domain id;
                            // a more complex one is needed for multiple domains
                            int tag = (my_address << 5) + rank; // WARN: maximum rank = 2^5 - 1
                            extended_domain_id_type id{static_cast<int>(rank), static_cast<address_type>(rank), tag};
                            std::vector<index_type> remote_index{};
                            remote_index.resize(send_counts[rank]);
                            std::memcpy(&remote_index[0],
                                    &send_indexes[static_cast<std::size_t>(send_displs[rank])],
                                    static_cast<std::size_t>(send_counts[rank]) * sizeof(index_type));
                            index_container_type ic{ {static_cast<int>(rank), std::move(remote_index)} };
                            p.send_halos().insert(std::make_pair(id, ic));
                        }
                    }

                    // update patterns list
                    my_patterns.push_back(p);

                }

                return pattern_container<P, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

            }

        };

    } // namespace detail

} // namespace gridtools

#endif /* INCLUDED_UNSTRUCTURED_PATTERN_HPP */
