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
#include <set>
#include <map>
#include <numeric>
//#include <cstring>
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
         * @tparam Index index type for domain (local indices) and iteration space
         * @tparam DomainId domain id type*/
        template<typename Communicator, typename Index, typename DomainId>
        class pattern<Communicator, unstructured::detail::grid<Index>, DomainId> {

            public:

                // member types
                using communicator_type = Communicator;
                using index_type = Index;
                using domain_id_type = DomainId;
                using address_type = typename communicator_type::address_type;
                using grid_type = unstructured::detail::grid<index_type>;
                using pattern_container_type = pattern_container<communicator_type, grid_type, domain_id_type>;

                friend class pattern_container<communicator_type, grid_type, domain_id_type>;

                /** @brief unstructured iteration space for accessing halo elements*/
                class iteration_space {

                    public:

                        using allocator_type = gridtools::ghex::allocator::cuda::unified_memory_allocator<index_type>;
                        using local_indices_type = std::vector<index_type, allocator_type>;

                    private:

                        local_indices_type m_local_indices;

                    public:

                        // ctors
                        iteration_space() noexcept = default;
                        iteration_space(const local_indices_type& local_indices) : m_local_indices{local_indices} {}

                        // member functions
                        std::size_t size() const noexcept { return m_local_indices.size(); }
                        const local_indices_type& local_indices() const noexcept { return m_local_indices; }
                        void push_back(const index_type idx) { m_local_indices.push_back(idx); }

                        // print
                        /** @brief print */
                        template <typename CharT, typename Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const iteration_space& is) {
                            os << "size = " << is.size() << ";\n"
                               << "local indices: [ ";
                            for (auto idx : is.local_indices()) { os << idx << " "; }
                            os << "]\n";
                            return os;
                        }

                };

                using index_container_type = std::vector<iteration_space>; // TO DO: should be simplified, just one halo per local domain

                /** @brief extended domain id, including rank, address and tag information*/
                struct extended_domain_id_type {

                    // members
                    domain_id_type id;
                    int mpi_rank;
                    address_type address;
                    int tag;

                    // member functions
                    /** @brief unique ordering given by address and tag*/
                    bool operator < (const extended_domain_id_type& other) const noexcept {
                        return address < other.address ? true : (address == other.address ? (tag < other.tag) : false);
                    }

                    /** @brief print*/
                    template <typename CharT, typename Traits>
                    friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const extended_domain_id_type& ext_id) {
                        os << "{id = " << ext_id.id
                           << ", rank = " << ext_id.mpi_rank
                           << ", address = " << ext_id.address
                           << ", tag = " << ext_id.tag << "}";
                        return os;
                    }

                };

                // halo map type
                using map_type = std::map<extended_domain_id_type, index_container_type>;

                // static member functions
                /** @brief compute number of elements in an object of type index_container_type*/
                static std::size_t num_elements(const index_container_type& c) noexcept {
                    std::size_t s{0};
                    for (const auto& is : c) s += is.size();
                    return s;
                }

            private:

                // members
                extended_domain_id_type m_id;
                map_type m_send_map;
                map_type m_recv_map;
                pattern_container_type* m_container;

            public:

                // ctors
                pattern(const extended_domain_id_type& id) :
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

            /** @brief constructs the pattern with the help of all to all communications*/
            template<typename Index>
            struct make_pattern_impl<unstructured::detail::grid<Index>> {

                template<typename Transport, typename ThreadPrimitives, typename HaloGenerator, typename DomainRange>
                static auto apply(tl::context<Transport, ThreadPrimitives>& context, HaloGenerator&& hgen, DomainRange&& d_range) {

                    // typedefs
                    using grid_type = unstructured::detail::grid<Index>;
                    using context_type = tl::context<Transport, ThreadPrimitives>;
                    using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
                    using communicator_type = typename context_type::communicator_type;
                    using domain_id_type = typename domain_type::domain_id_type;
                    using global_index_type = typename domain_type::global_index_type;
                    using pattern_type = pattern<communicator_type, grid_type, domain_id_type>;
                    using index_type = typename pattern_type::index_type;
                    using address_type = typename pattern_type::address_type;
                    using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
                    using iteration_space_type = typename pattern_type::iteration_space;
                    using index_container_type = typename pattern_type::index_container_type;
                    using vertices_type = std::vector<global_index_type>;
                    using vertices_map_type = std::map<global_index_type, index_type>;
                    using rank_local_index_type = index_type;
                    using allocator_type = typename iteration_space_type::allocator_type;

                    // get setup comm and new comm, and then this rank, this address and size from new comm
                    auto comm = context.get_setup_communicator();
                    auto new_comm = context.get_serial_communicator();
                    auto my_rank = new_comm.rank();
                    auto my_address = new_comm.address();
                    auto size = new_comm.size();

                    // setup patterns
                    std::vector<pattern_type> my_patterns;
                    for (const auto& d : d_range) {
                        pattern_type p{{d.domain_id(), my_rank, my_address, 0}};
                        my_patterns.push_back(p);
                    }

                    /*
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
                    */

                    // gather halos from all local domains on all ranks
                    int num_domains = static_cast<int>(d_range.size()); // number of local domains (int, since has to be used as elem counter)
                    auto all_num_domains = comm.all_gather(num_domains).get(); // numbers of local domains on all ranks
                    std::vector<domain_id_type> domain_ids{}; // domain id for each local domain
                    std::vector<std::size_t> halo_sizes{}; // halo size for each local domain
                    vertices_type reduced_halo{}; // single reduced halo with halo vertices of all local domains
                    for (const auto& d : d_range) {
                        domain_ids.push_back(d.domain_id());
                        auto h = hgen(d);
                        halo_sizes.push_back(h.size());
                        reduced_halo.insert(reduced_halo.end(), h.vertices.begin(), h.vertices.end());
                    }
                    auto all_domain_ids = comm.all_gather(domain_ids, all_num_domains).get(); // domain id for each local domain on all ranks
                    auto all_halo_sizes = comm.all_gather(halo_sizes, all_num_domains).get(); // halo size for each local domain on all ranks
                    std::vector<int> all_reduced_halo_sizes{}; // size of reduced halo on all ranks (int, since has to be used as elem counter)
                    for (const auto& hs : all_halo_sizes) {
                        all_reduced_halo_sizes.push_back(static_cast<int>(std::accumulate(hs.begin(), hs.end(), 0)));
                    }
                    auto all_reduced_halos = comm.all_gather(reduced_halo, all_reduced_halo_sizes).get(); // single reduced halo with halo vertices of all local domains on all ranks

                    // other setup helpers
                    auto all_addresses = comm.all_gather(my_address).get(); // addresses of all ranks
                    std::vector<domain_id_type> max_domain_ids{}; // max domain id on every rank
                    for (const auto& d_ids : all_domain_ids) {
                        max_domain_ids.push_back(std::max_element(d_ids.begin(), d_ids.end()));
                    }
                    domain_id_type max_domain_id = std::max_element(max_domain_ids.begin(), max_domain_ids.end()); // max domain id among all ranks
                    int m_max_tag = (max_domain_id << 7) + max_domain_id; // TO DO: maximum shift should not be hard-coded. TO DO: should add 1?

                    // ========== SEND ==========

                    std::vector<std::vector<int>> all_send_counts{}; // number of elements to be sent from each local domain to all ranks (int, since has to be used as elem counter)
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        std::vector<int> other_send_counts{};
                        other_send_counts.resize(all_num_domains[other_rank]);
                        std::fill(other_send_counts.begin(), other_send_counts.end(), 0);
                        all_send_counts.push_back(std::move(other_send_counts)); // TO DO: better way to achieve that
                    }
                    std::vector<std::vector<rank_local_index_type>> all_send_indices{}; // elements to be sent from all local domains to all ranks (in terms of rank local indices)
                    all_send_indices.resize(size);

                    for (auto p = 0; p < my_patterns.size(); ++p) { // loop through local domains

                        auto d = d_range[p]; // local domain
                        auto my_id = d.domain_id(); // local domain id
                        vertices_map_type d_vertices_map{}; // local vertices map
                        for (auto local_idx = 0; local_idx < d.inner_size(); ++local_idx) {
                            d_vertices_map.insert(std::make_pair(d.vertices()[local_idx], local_idx));
                        }

                        std::size_t all_reduced_halos_start_idx{0};
                        for (auto other_rank = 0; other_rank < size; ++other_rank) { // loop through all_reduced_halos, one rank at a time
                            auto other_address = all_addresses[other_rank];
                            rank_local_index_type other_rank_idx{0};
                            for (auto other_domain_idx = 0; other_domain_idx < all_num_domains[other_rank]; ++other_domain_idx) { // loop through all domains on other rank; TO DO: std::size_t?
                                auto other_halo_size = all_halo_sizes[other_rank][other_domain_idx];
                                if (other_halo_size) {
                                    auto other_id = all_domain_ids[other_rank][other_domain_idx];
                                    int tag = (static_cast<int>(my_id) << 7) + static_cast<int>(other_id); // TO DO: maximum shift should not be hard-coded
                                    extended_domain_id_type id{other_id, other_rank, other_address, tag};
                                    iteration_space_type is{};
                                    for (auto all_reduced_halos_idx = all_reduced_halos_start_idx;
                                         all_reduced_halos_idx < all_reduced_halos_start_idx + other_halo_size;
                                         ++all_reduced_halos_idx, ++other_rank_idx) { // loop through halo vertices
                                        auto it = d_vertices_map.find(all_reduced_halos[all_reduced_halos_idx]);
                                        if (it != d_vertices_map.end()) {
                                            is.push_back((*it).second);
                                            all_send_indices[other_rank].push_back(other_rank_idx);
                                        }
                                    }
                                    index_container_type ic{is};
                                    my_patterns[p].send_halos().insert(std::make_pair(id, ic));
                                    all_send_counts[other_rank][p] += static_cast<int>(is.size());
                                    all_reduced_halos_start_idx += other_halo_size;
                                }
                            }
                        }

                    }

                    // FROM HERE

                    // setup all-to-all communication, send side (TO DO: all_to_all interface chould be made similar to all_gather, this will avoid vector flattening)
                    std::vector<int> all_flat_send_counts{};
                    std::vector<int> all_flat_send_counts_displs{};
                    all_flat_send_counts_displs.resize(static_cast<std::size_t>(size));
                    all_flat_send_counts_displs[0] = 0;
                    std::vector<int> all_rank_send_counts;
                    all_rank_send_counts.resize(static_cast<std::size_t>(size));
                    std::vector<int> all_rank_send_displs;
                    all_rank_send_displs.resize(static_cast<std::size_t>(size));
                    all_rank_send_displs[0] = 0;
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        int total_rank_count{0};
                        for (auto sc : all_send_counts[static_cast<std::size_t>(other_rank)]) {
                            all_flat_send_counts.push_back(sc);
                            total_rank_count += sc;
                        }
                        all_rank_send_counts[static_cast<std::size_t>(other_rank)] = total_rank_count;
                        if (other_rank > 0) {
                            all_flat_send_counts_displs[static_cast<std::size_t>(other_rank)] =
                                    all_flat_send_counts_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    all_num_domains[static_cast<std::size_t>(other_rank - 1)];
                            all_rank_send_displs[static_cast<std::size_t>(other_rank)] =
                                    all_rank_send_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    total_rank_count;
                        }
                    }

                    // ========== RECV ==========

                    // setup all-to-all communication, recv side
                    std::vector<int> all_flat_recv_counts{};
                    all_flat_recv_counts.resize(size);
                    comm.all_to_allv(all_flat_send_counts, all_num_domains, all_flat_send_counts_displs,
                                     all_flat_recv_counts, all_num_domains, all_flat_send_counts_displs);

                    // setup receive halos
                    // comm.all_to_allv();
                    // USE: - all_send_indices

                    /*
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
                    */

                    return pattern_container<communicator_type, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

                }

            };

        } // namespace detail

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_UNSTRUCTURED_PATTERN_HPP */
