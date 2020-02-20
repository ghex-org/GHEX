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
#ifndef INCLUDED_GHEX_UNSTRUCTURED_PATTERN_HPP
#define INCLUDED_GHEX_UNSTRUCTURED_PATTERN_HPP

#include <vector>
#include <map>
#include <numeric>
#include <cstring>
#include <iosfwd>

#include "../transport_layer/context.hpp"
#include "../allocator/unified_memory_allocator.hpp"
#include "../pattern.hpp"
#include "../buffer_info.hpp"
#include "./grid.hpp"


namespace gridtools {
    namespace ghex {

        /** @brief unstructured pattern specialization
         *
         * This class provides access to the receive and send iteration spaces, determined by the halos,
         * and holds all connections to the neighbors.
         *
         * @tparam Communicator communicator type
         * @tparam Index index type for domain and iteration space
         * @tparam DomainId domain id type*/
        template<typename Communicator, typename Index, typename DomainId>
        class pattern<Communicator, unstructured::detail::grid<Index>, DomainId> {

            public:

                // member types
                using communicator_type = Communicator;
                using address_type = typename communicator_type::address_type;
                using index_type = Index;
                using grid_type = unstructured::detail::grid<index_type>;
                using domain_id_type = DomainId;
                using pattern_container_type = pattern_container<communicator_type, grid_type, domain_id_type>;

                friend class pattern_container<communicator_type, grid_type, domain_id_type>;

                /** @brief unstructured iteration space for accessing halo elements
                 * Indices in 'm_local_index' are always assumed as local to partition 'm_partition'
                 * (note that, on the other hand, each element returned by a halo generator has two series of indices,
                 * one local to the receiving partition ('local index') and one local to the remote (sending) partiton ('remote index').
                 * Number of levels is provided as well, with the assumption that each 2D element is a column
                 * with 'levels' vertical elements (default is 1, i.e. 2D or fully unstructured case)*/
                class iteration_space {

                    public:

                        using u_m_allocator_t = gridtools::ghex::allocator::cuda::unified_memory_allocator<index_type>;

                    private:

                        int m_partition;
                        std::vector<index_type, u_m_allocator_t> m_local_index;
                        std::size_t m_levels;

                    public:

                        // ctors
                        iteration_space() noexcept = default;
                        iteration_space(const int partition,
                                const std::vector<index_type, u_m_allocator_t>& local_index,
                                const std::size_t levels = 1) :
                            m_partition{partition},
                            m_local_index{local_index},
                            m_levels{levels} {}
                        iteration_space(const int partition, const index_type first, const index_type last, const std::size_t levels = 1) :
                            m_partition{partition},
                            m_local_index{},
                            m_levels{levels} {
                            m_local_index.resize(static_cast<std::size_t>(last - first + 1));
                            for (index_type idx = first; idx <= last; ++idx) {
                                m_local_index[static_cast<std::size_t>(idx)] = idx;
                            }
                        }

                        // member functions
                        int partition() const noexcept { return m_partition; }
                        const std::vector<index_type, u_m_allocator_t>& local_index() const noexcept { return m_local_index; }
                        std::size_t levels() const noexcept { return m_levels; }
                        std::size_t size() const noexcept { return m_local_index.size() * m_levels; }

                        // print
                        /** @brief print */
                        template<class CharT, class Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const iteration_space& is) {
                            os << "size = " << is.size() << ";\n"
                               << "# levels = " << is.levels() << ";\n"
                               << "partition = " << is.partition() << ";\n"
                               << "local indexes: [ ";
                            for (auto idx : is.local_index()) { os << idx << " "; }
                            os << "]\n";
                            return os;
                        }

                };

                using iteration_space_pair = iteration_space;
                using index_container_type = std::vector<iteration_space_pair>;

                /** @brief extended domain id, including rank, address and tag information
                 * WARN: domain id temporarily set equal to rank, differently from structured pattern case*/
                struct extended_domain_id_type {

                    // members
                    domain_id_type id;
                    int mpi_rank;
                    address_type address;
                    int tag;

                    // member functions
                    /** @brief unique ordering given by address and tag*/
                    bool operator < (const extended_domain_id_type& other) const noexcept {
                        return (address < other.address ? true : (address == other.address ? (tag < other.tag) : false));
                    }

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
                iteration_space_pair m_domain;
                extended_domain_id_type m_id;
                map_type m_send_map;
                map_type m_recv_map;
                pattern_container_type* m_container;

            public:

                // ctors
                pattern(const iteration_space_pair& domain, const extended_domain_id_type& id) :
                    m_domain{domain},
                    m_id{id},
                    m_send_map{},
                    m_recv_map{},
                    m_container{nullptr} {}
                pattern(const pattern&) = default;
                pattern(pattern&&) = default;

                // member functions
                domain_id_type domain_id() const noexcept { return m_id.id; }
                extended_domain_id_type extended_domain_id() const noexcept { return m_id; }
                map_type& send_halos() noexcept { return m_send_map; }
                const map_type& send_halos() const noexcept { return m_send_map; }
                map_type& recv_halos() noexcept { return m_recv_map; }
                const map_type& recv_halos() const noexcept { return m_recv_map; }
                const pattern_container_type& container() const noexcept { return *m_container; }

                /** @brief tie pattern to field
                 * @tparam Field field type
                 * @param field field instance
                 * @return buffer_info object which holds pointers to the field and the pattern*/
                template<typename Field>
                buffer_info<pattern, typename Field::arch_type, Field> operator()(Field& field) const {
                    return { *this, field, field.device_id() };
                }

        };

        namespace detail {

            /** @brief constructs the pattern with the help of all to all communications
             * Note: halos are decomposed into horizontal dimensions and vertical levels,
             * but a fully unstructured case can be treated exactly as a 2D case (m_levels = 1)*/
            template<typename Index>
            struct make_pattern_impl<unstructured::detail::grid<Index>> {

                template<typename Transport, typename ThreadPrimitives, typename HaloGenerator, typename DomainRange>
                static auto apply(tl::context<Transport, ThreadPrimitives>& context, HaloGenerator&& hgen, DomainRange&& d_range) {

                    // typedefs
                    using context_type = tl::context<Transport, ThreadPrimitives>;
                    using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
                    using domain_id_type = typename domain_type::domain_id_type;
                    using grid_type = unstructured::detail::grid<Index>;
                    using communicator_type = typename context_type::communicator_type;
                    using pattern_type = pattern<communicator_type, grid_type, domain_id_type>;
                    using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
                    using index_container_type = typename pattern_type::index_container_type;
                    using index_type = typename pattern_type::index_type;
                    using address_type = typename pattern_type::address_type;
                    using u_m_allocator_t = gridtools::ghex::allocator::cuda::unified_memory_allocator<index_type>;

                    // get this rank, address and size from new communicator
                    auto comm = context.get_setup_communicator();
                    auto new_comm = context.get_serial_communicator();
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
                    std::vector<std::size_t> recv_levels{};
                    recv_levels.resize(size);
                    std::vector<std::size_t> send_levels{};
                    send_levels.resize(size);

                    // needed with multiple domains per rank
                    int m_max_tag = 999;

                    for (const auto& d : d_range) { // WARN: so far, multiple domains are not fully supported

                        // setup pattern
                        pattern_type p{{my_rank, d.first(), d.last(), d.levels()}, {my_rank, my_rank, my_address, 0}};

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
                                int tag = (h.partition() << 7) + my_address; // WARN: maximum address / rank = 2^7 - 1
                                extended_domain_id_type id{h.partition(), h.partition(), static_cast<address_type>(h.partition()), tag}; // WARN: address is not obtained from the other domain
                                index_container_type ic{ {h.partition(), h.local_index(), h.levels()} };
                                p.recv_halos().insert(std::make_pair(id, ic));
                                recv_counts[static_cast<std::size_t>(h.partition())] = static_cast<int>(h.size());
                                recv_levels[static_cast<std::size_t>(h.partition())] = h.levels();
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
                        comm.all_to_all(recv_counts, send_counts);
                        comm.all_to_all(recv_levels, send_levels);
                        send_count = std::accumulate(send_counts.begin(), send_counts.end(), 0);
                        send_indexes.resize(send_count);
                        send_displs[0] = 0;
                        for (std::size_t rank = 1; rank < size; ++rank) {
                            send_displs[rank] = send_displs[rank - 1] + send_counts[rank - 1];
                        }

                        // set up send halos
                        comm.all_to_allv(recv_indexes, recv_counts, recv_displs,
                                send_indexes, send_counts, send_displs);
                        for (std::size_t rank = 0; rank < size; ++rank) {
                            if (send_counts[rank]) {
                                // WARN: very simplified definition of extended domain id;
                                // a more complex one is needed for multiple domains
                                int tag = (my_address << 7) + rank; // WARN: maximum rank / address = 2^7 - 1
                                extended_domain_id_type id{static_cast<int>(rank), static_cast<int>(rank), static_cast<address_type>(rank), tag};
                                std::vector<index_type, u_m_allocator_t> remote_index{};
                                remote_index.resize(send_counts[rank]);
                                std::memcpy(&remote_index[0],
                                        &send_indexes[static_cast<std::size_t>(send_displs[rank])],
                                        static_cast<std::size_t>(send_counts[rank]) * sizeof(index_type));
                                index_container_type ic{ {static_cast<int>(rank), std::move(remote_index), send_levels[rank]} };
                                p.send_halos().insert(std::make_pair(id, ic));
                            }
                        }

                        // update patterns list
                        my_patterns.push_back(p);

                    }

                    return pattern_container<communicator_type, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

                }

            };

        } // namespace detail

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_UNSTRUCTURED_PATTERN_HPP */
