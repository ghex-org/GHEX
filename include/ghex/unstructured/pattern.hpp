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
#include <algorithm>
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
                        std::size_t m_levels;

                    public:

                        // ctors
                        iteration_space(const std::size_t levels = 1) noexcept : m_levels{levels} {}
                        iteration_space(const local_indices_type& local_indices,
                                        const std::size_t levels = 1) :
                            m_local_indices{local_indices},
                            m_levels{levels} {}

                        // member functions
                        std::size_t size() const noexcept { return m_local_indices.size(); }
                        std::size_t levels() const noexcept { return m_levels; }
                        const local_indices_type& local_indices() const noexcept { return m_local_indices; }
                        void push_back(const index_type idx) { m_local_indices.push_back(idx); }

                        // print
                        /** @brief print */
                        template <typename CharT, typename Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const iteration_space& is) {
                            os << "size = " << is.size() << ";\n"
                               << "levels = " << is.levels() << ";\n"
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
                           << ", tag = " << ext_id.tag << "}\n";
                        return os;
                    }

                };

                // halo map type
                using map_type = std::map<extended_domain_id_type, index_container_type>;

                // static member functions
                /** @brief compute number of elements in an object of type index_container_type*/
                static std::size_t num_elements(const index_container_type& c) noexcept {
                    std::size_t s{0};
                    for (const auto& is : c) s += (is.size() * is.levels());
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

                /** @brief specialization used when no hints on neighbor domains are provided
                 * The workflow is as follows:
                 * - all gather communications to retrive receive halos from all domains, plus some metadata;
                 * - for each local domain, loop through all receive halos to fetch items to be sent to each other domian,
                 *   and set up send halos in pattern, as well as vector of local indices to be sent to other domains;
                 * - all to all communication to inform each other domain of the indices which will be sent,
                 *   which becomes receive indices on the receive side (2 all to all communications in total,
                 *   one for the send / recv elements counters and one for the send / recv indices);
                 * - reconstruct recv halos on the receive side and set up receive halos in pattern.*/
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
                    using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
                    using iteration_space_type = typename pattern_type::iteration_space;
                    using index_container_type = typename pattern_type::index_container_type;
                    using vertices_type = std::vector<global_index_type>;
                    using vertices_map_type = std::map<global_index_type, index_type>;

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

                    // gather halos from all local domains on all ranks. TO DO: from here to the end of the function, are all casts actually needed?
                    int num_domains = static_cast<int>(d_range.size()); // number of local domains (int, since has to be used as elem counter)
                    auto all_num_domains = comm.all_gather(num_domains).get(); // numbers of local domains on all ranks
                    std::vector<domain_id_type> domain_ids{}; // domain id for each local domain
                    std::vector<std::size_t> halo_sizes{}; // halo size for each local domain
                    std::vector<std::size_t> num_levels{}; // halo levels for each local domain
                    vertices_type reduced_halo{}; // single reduced halo with halo vertices of all local domains
                    for (const auto& d : d_range) {
                        domain_ids.push_back(d.domain_id());
                        auto h = hgen(d);
                        halo_sizes.push_back(h.size());
                        num_levels.push_back(h.levels());
                        reduced_halo.insert(reduced_halo.end(), h.vertices().begin(), h.vertices().end());
                    }
                    auto all_domain_ids = comm.all_gather(domain_ids, all_num_domains).get(); // domain id for each local domain on all ranks
                    auto all_halo_sizes = comm.all_gather(halo_sizes, all_num_domains).get(); // halo size for each local domain on all ranks
                    std::vector<int> all_reduced_halo_sizes{}; // size of reduced halo on all ranks (int, since has to be used as elem counter)
                    for (const auto& hs : all_halo_sizes) {
                        all_reduced_halo_sizes.push_back(static_cast<int>(std::accumulate(hs.begin(), hs.end(), 0)));
                    }
                    auto all_reduced_halos = comm.all_gather(reduced_halo, all_reduced_halo_sizes).get(); // single reduced halos with halo vertices of all local domains on all ranks

                    // other setup helpers
                    auto all_addresses = comm.all_gather(my_address).get(); // addresses of all ranks
                    std::vector<domain_id_type> max_domain_ids{}; // max domain id on every rank
                    for (const auto& d_ids : all_domain_ids) {
                        max_domain_ids.push_back(*(std::max_element(d_ids.begin(), d_ids.end())));
                    }
                    domain_id_type max_domain_id = *(std::max_element(max_domain_ids.begin(), max_domain_ids.end())); // max domain id among all ranks
                    int m_max_tag = (max_domain_id << 7) + max_domain_id; // TO DO: maximum shift should not be hard-coded. TO DO: should add 1?

                    // ========== SEND ==========

                    std::vector<std::vector<int>> all_send_counts(size); // number of elements to be sent from each local domain to all ranks (int, since has to be used as elem counter)
                    for (auto& scs : all_send_counts) {
                        scs.resize(d_range.size());
                    }

                    std::vector<std::vector<index_type>> all_send_indices(size); // elements to be sent from all local domains to all ranks (in terms of local indices)

                    for (std::size_t p = 0; p < my_patterns.size(); ++p) { // loop through local domains

                        auto d = d_range[p]; // local domain
                        auto my_id = d.domain_id(); // local domain id
                        vertices_map_type d_vertices_map{}; // local vertices map
                        for (std::size_t local_idx = 0; local_idx < d.inner_size(); ++local_idx) {
                            d_vertices_map.insert(std::make_pair(d.vertices()[local_idx], local_idx));
                        }

                        for (auto other_rank = 0; other_rank < size; ++other_rank) { // loop through all_reduced_halos, one rank at a time
                            auto other_address = all_addresses[static_cast<std::size_t>(other_rank)];
                            std::size_t reduced_halo_start_idx{0};
                            index_type rank_local_idx{0};
                            for (auto other_domain_idx = 0; other_domain_idx < all_num_domains[static_cast<std::size_t>(other_rank)]; ++other_domain_idx) { // loop through all domains on other rank; TO DO: std::size_t?
                                auto other_halo_size = all_halo_sizes[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)];
                                if (other_halo_size) {
                                    auto other_id = all_domain_ids[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)];
                                    int tag = (static_cast<int>(my_id) << 7) + static_cast<int>(other_id); // TO DO: maximum shift should not be hard-coded
                                    extended_domain_id_type id{other_id, other_rank, other_address, tag};
                                    iteration_space_type is{num_levels[p]};
                                    for (auto reduced_halo_idx = reduced_halo_start_idx;
                                         reduced_halo_idx < reduced_halo_start_idx + other_halo_size;
                                         ++reduced_halo_idx, ++rank_local_idx) { // loop through halo vertices
                                        auto it = d_vertices_map.find(all_reduced_halos[static_cast<std::size_t>(other_rank)][reduced_halo_idx]);
                                        if (it != d_vertices_map.end()) {
                                            is.push_back((*it).second);
                                            all_send_indices[static_cast<std::size_t>(other_rank)].push_back(rank_local_idx);
                                        }
                                    }
                                    if (is.size()) {
                                        index_container_type ic{is};
                                        my_patterns[p].send_halos().insert(std::make_pair(id, ic));
                                        all_send_counts[static_cast<std::size_t>(other_rank)][p] += static_cast<int>(is.size());
                                    }
                                    reduced_halo_start_idx += other_halo_size;
                                }
                            }
                        }

                    }

                    // setup all-to-all communications, send side (TO DO: all_to_all interface should be made similar to all_gather, this will avoid vector flattening)
                    // first communication variables setup
                    std::vector<int> all_flat_send_counts{}; // 1/6
                    std::vector<int> all_my_num_domains(size); // 2/6
                    std::fill(all_my_num_domains.begin(), all_my_num_domains.end(), num_domains);
                    std::vector<int> all_my_num_domains_displs(size); // 3/6
                    all_my_num_domains_displs[0] = 0;
                    // 4/6: recv side
                    // 5/6: all_num_domains
                    std::vector<int> all_num_domains_displs(size); // 6/6
                    all_num_domains_displs[0] = 0;
                    // second communication variables setup
                    std::vector<index_type> all_flat_send_indices; // 1/6
                    std::vector<int> all_rank_send_counts(size); // 2/6
                    std::vector<int> all_rank_send_displs(size); // 3/6
                    all_rank_send_displs[0] = 0;
                    // 4/6: recv side
                    // 5/6: recv side, obtained from previous communication
                    // 6/6: recv side, obtained from 5/6
                    // other setup
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        all_flat_send_counts.insert(all_flat_send_counts.end(),
                                                    all_send_counts[static_cast<std::size_t>(other_rank)].begin(),
                                                    all_send_counts[static_cast<std::size_t>(other_rank)].end());
                        all_flat_send_indices.insert(all_flat_send_indices.end(),
                                                     all_send_indices[static_cast<std::size_t>(other_rank)].begin(),
                                                     all_send_indices[static_cast<std::size_t>(other_rank)].end());
                        all_rank_send_counts[static_cast<std::size_t>(other_rank)] =
                                static_cast<int>(all_send_indices[static_cast<std::size_t>(other_rank)].size());
                        if (other_rank > 0) {
                            all_my_num_domains_displs[static_cast<std::size_t>(other_rank)] =
                                    all_my_num_domains_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    num_domains;
                            all_num_domains_displs[static_cast<std::size_t>(other_rank)] =
                                    all_num_domains_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    all_num_domains[static_cast<std::size_t>(other_rank - 1)];
                            all_rank_send_displs[static_cast<std::size_t>(other_rank)] =
                                    all_rank_send_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    all_rank_send_counts[static_cast<std::size_t>(other_rank - 1)];
                        }
                    }

                    // ========== RECV ==========

                    // setup all-to-all communications, recv side
                    // first communication variables setup
                    auto tot_num_domains = std::accumulate(all_num_domains.begin(), all_num_domains.end(), 0); // overall number of local domains
                    std::vector<int> all_flat_recv_counts(tot_num_domains); // 4/6
                    // first communication
                    comm.all_to_allv(all_flat_send_counts, all_my_num_domains, all_my_num_domains_displs,
                                     all_flat_recv_counts, all_num_domains, all_num_domains_displs);
                    // second communication variables setup
                    auto tot_recv_count = std::accumulate(all_flat_recv_counts.begin(), all_flat_recv_counts.end(), 0); // overall number of received indices
                    std::vector<index_type> all_flat_recv_indices(tot_recv_count); // 4/6
                    std::vector<int> all_rank_recv_counts(size); // 5/6
                    std::fill(all_rank_recv_counts.begin(), all_rank_recv_counts.end(), 0);
                    std::vector<int> all_rank_recv_displs(size); // 6/6
                    all_rank_recv_displs[0] = 0;
                    // other setup
                    std::size_t all_flat_recv_counts_idx{0};
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        for (auto other_domain_idx = 0; other_domain_idx < all_num_domains[static_cast<std::size_t>(other_rank)]; ++other_domain_idx) {
                            all_rank_recv_counts[static_cast<std::size_t>(other_rank)] += all_flat_recv_counts[all_flat_recv_counts_idx++];
                        }
                        if (other_rank > 0) {
                            all_rank_recv_displs[static_cast<std::size_t>(other_rank)] =
                                    all_rank_recv_displs[static_cast<std::size_t>(other_rank - 1)] +
                                    all_rank_recv_counts[static_cast<std::size_t>(other_rank - 1)];
                        }
                    }
                    // second communication
                    comm.all_to_allv(all_flat_send_indices, all_rank_send_counts, all_rank_send_displs,
                                     all_flat_recv_indices, all_rank_recv_counts, all_rank_recv_displs);

                    // back to multidimensional objects
                    std::vector<std::vector<int>> all_recv_counts(size); // number of elements to be received from each local domain from all ranks (int, since will be derived from elem counter)
                    std::vector<std::vector<index_type>> all_recv_indices(size);
                    all_flat_recv_counts_idx = 0;
                    std::size_t all_flat_recv_indices_start_idx{0};
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        for (auto other_domain_idx = 0; other_domain_idx < all_num_domains[static_cast<std::size_t>(other_rank)]; ++other_domain_idx) {
                            all_recv_counts[static_cast<std::size_t>(other_rank)].push_back(all_flat_recv_counts[all_flat_recv_counts_idx++]);
                            for (auto all_flat_recv_indices_idx = all_flat_recv_indices_start_idx;
                                 all_flat_recv_indices_idx < all_flat_recv_indices_start_idx +
                                 static_cast<std::size_t>(all_recv_counts[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)]);
                                 ++all_flat_recv_indices_idx) {
                                all_recv_indices[static_cast<std::size_t>(other_rank)].push_back(all_flat_recv_indices[all_flat_recv_indices_idx]);
                            }
                            all_flat_recv_indices_start_idx +=
                                    static_cast<std::size_t>(all_recv_counts[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)]);
                        }
                    }

                    index_type rank_local_start_index{0};
                    for (std::size_t p = 0; p < my_patterns.size(); ++p) { // loop through local domains. TO DO: this should probably be the innermost loop

                        auto d = d_range[p]; // local domain
                        auto my_id = d.domain_id(); // local domain id
                        auto halo_size = hgen(d).size(); // local halo size

                        for (auto other_rank = 0; other_rank < size; ++other_rank) {
                            auto other_address = all_addresses[static_cast<std::size_t>(other_rank)];
                            std::size_t recv_indices_start_idx{0};
                            for (auto other_domain_idx = 0; other_domain_idx < all_num_domains[static_cast<std::size_t>(other_rank)]; ++other_domain_idx) {
                                if (all_recv_counts[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)]) {
                                    auto other_id = all_domain_ids[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)];
                                    int tag = (static_cast<int>(other_id) << 7) + static_cast<int>(my_id); // TO DO: maximum shift should not be hard-coded
                                    extended_domain_id_type id{other_id, other_rank, other_address, tag};
                                    iteration_space_type is{num_levels[p]};
                                    for (auto recv_indices_idx = recv_indices_start_idx;
                                         recv_indices_idx < recv_indices_start_idx +
                                         static_cast<std::size_t>(all_recv_counts[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)]);
                                         ++recv_indices_idx) {
                                        if ((all_recv_indices[static_cast<std::size_t>(other_rank)][recv_indices_idx] >=
                                             rank_local_start_index) &&
                                                (all_recv_indices[static_cast<std::size_t>(other_rank)][recv_indices_idx] <
                                                 rank_local_start_index + static_cast<index_type>(halo_size))) {
                                            is.push_back(all_recv_indices[static_cast<std::size_t>(other_rank)][recv_indices_idx] -
                                                    rank_local_start_index +
                                                    d.inner_size()); // index offset
                                        }
                                    }
                                    if (is.size()) {
                                        index_container_type ic{is};
                                        my_patterns[p].recv_halos().insert(std::make_pair(id, ic));
                                    }
                                    recv_indices_start_idx +=
                                            static_cast<std::size_t>(all_recv_counts[static_cast<std::size_t>(other_rank)][static_cast<std::size_t>(other_domain_idx)]);
                                }
                            }
                        }

                        rank_local_start_index += halo_size;
                    }

                    return pattern_container<communicator_type, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

                }

                /** @brief specialization used when receive domain ids generator is provided
                 * The workflow is as follows:
                 * - all gather communications to retrive some metadata;
                 * - set up receive halos in pattern, as well as vector of local indices to be received from other domains.
                 *   Note: recv halos are set using the information provided by the user on the receive domain ids;
                 * - all to all communication to inform each other domain of the indices which will be received,
                 *   which becomes send indices on the send side (2 all to all communications in total,
                 *   one for the recv / send elements counters and one for the recv / send indices);
                 * - reconstruct send halos on the send side and set up send halos in pattern.*/
                template<typename Transport, typename ThreadPrimitives, typename HaloGenerator, typename RecvDomainIdsGen, typename DomainRange>
                static auto apply(tl::context<Transport, ThreadPrimitives>& context, HaloGenerator&& hgen, RecvDomainIdsGen&& recv_domain_ids_gen, DomainRange&& d_range) {

                    // typedefs
                    using grid_type = unstructured::detail::grid<Index>;
                    using context_type = tl::context<Transport, ThreadPrimitives>;
                    using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
                    using communicator_type = typename context_type::communicator_type;
                    using domain_id_type = typename domain_type::domain_id_type;
                    using pattern_type = pattern<communicator_type, grid_type, domain_id_type>;
                    using index_type = typename pattern_type::index_type;
                    using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
                    using iteration_space_type = typename pattern_type::iteration_space;
                    using index_container_type = typename pattern_type::index_container_type;

                    using all_recv_counts_type = std::vector<std::map<domain_id_type, std::vector<int>>>;
                    using all_send_counts_type = std::vector<std::map<domain_id_type, std::vector<int>>>;
                    using all_recv_indices_type = std::vector<std::map<domain_id_type, std::vector<index_type>>>;

                    // get setup comm and new comm, and then this rank, this address and size from new comm
                    auto comm = context.get_setup_communicator();
                    auto new_comm = context.get_serial_communicator();
                    auto my_rank = new_comm.rank();
                    auto my_address = new_comm.address();
                    auto size = new_comm.size();

                    // gather domain ids and addresses for all ranks
                    int num_domains = static_cast<int>(d_range.size()); // number of local domains (int, since has to be used as elem counter)
                    auto all_num_domains = comm.all_gather(num_domains).get(); // numbers of local domains on all ranks
                    std::map<domain_id_type, std::size_t> local_domain_ids_map{}; // map between local domain ids and indices in d_range
                    for (std::size_t idx = 0; idx < d_range.size(); ++idx) {
                        local_domain_ids_map.insert({d_range[idx].domain_id(), idx});
                    }
                    std::vector<domain_id_type> domain_ids(d_range.size()); // domain id for each local domain (ordered)
                    std::transform(local_domain_ids_map.begin(), local_domain_ids_map.end(), domain_ids.begin(), [](auto p){return p.first;});
                    auto all_domain_ids = comm.all_gather(domain_ids, all_num_domains).get(); // domain id for each local domain on all ranks
                    auto all_addresses = comm.all_gather(my_address).get(); // addresses of all ranks

                    // set max tag
                    auto max_domain_id = comm.max_element(domain_ids);
                    int m_max_tag = (max_domain_id << 7) + max_domain_id; // max tag TO DO: maximum shift should not be hard-coded. TO DO: should add 1?

                    // helpers
                    std::map<domain_id_type, int> domain_id_to_rank; // helper domain_id to rank map (will be filled only with needed domain ids)

                    // recv side setup
                    all_recv_counts_type all_recv_counts(size);
                    for (auto other_rank = 0; other_rank < size; ++other_rank) {
                        for (auto id : all_domain_ids[other_rank]) {
                            all_recv_counts[other_rank].insert({id, std::vector<int>(d_range.size(), 0)});
                        }
                    }
                    all_recv_indices_type all_recv_indices(size);

                    // setup patterns, with only recv halos for now
                    std::vector<pattern_type> my_patterns;
                    std::size_t d_reidx{0};
                    for (const auto& d_id_idx : local_domain_ids_map) { // sorted by domain id
                        const auto& d = d_range[d_id_idx.second];
                        pattern_type p{{d.domain_id(), my_rank, my_address, 0}}; // construct pattern
                        std::map<domain_id_type, iteration_space_type> tmp_is_map{};// local helper for filling iteration spaces
                        auto h = hgen(d);
                        auto r_ids = recv_domain_ids_gen(d);
                        for (std::size_t h_idx = 0; h_idx < h.size(); ++h_idx) {
                            domain_id_to_rank.insert({r_ids.domain_ids()[h_idx], r_ids.ranks()[h_idx]});
                            tmp_is_map[r_ids.domain_ids()[h_idx]].push_back(h.local_indices()[h_idx]);
                            all_recv_counts[r_ids.ranks()[h_idx]].at(r_ids.domain_ids()[h_idx])[d_reidx] += 1; // OK
                            all_recv_indices[r_ids.ranks()[h_idx]][r_ids.domain_ids()[h_idx]].push_back(r_ids.remote_indices()[h_idx]); // OK
                        }
                        for (const auto& d_id_is : tmp_is_map) {
                            auto other_id = d_id_is.first;
                            auto other_rank = domain_id_to_rank.at(other_id);
                            auto other_address = all_addresses[other_rank];
                            int tag = (static_cast<int>(other_id) << 7) + static_cast<int>(d.domain_id()); // TO DO: maximum shift should not be hard-coded
                            extended_domain_id_type id{other_id, other_rank, other_address, tag};
                            iteration_space_type is{d_id_is.second.local_indices(), h.levels()};
                            index_container_type ic{is};
                            p.recv_halos().insert({id, ic});
                        }
                        my_patterns.push_back(p);
                        ++d_reidx;
                    }

                    // Setup first all-to-all (1-6): recv / send counts
                    std::vector<int> all_flat_recv_counts{}; // 1/6
                    for (const auto& rc_map : all_recv_counts) {
                        for (const auto& rc_pair : rc_map) {
                            all_flat_recv_counts.insert(all_flat_recv_counts.end(), rc_pair.second.begin(), rc_pair.second.end());
                        }
                    }
                    std::vector<int> all_num_counts(size); // same for send and recv // 2/6
                    std::transform(all_num_domains.begin(), all_num_domains.end(), all_num_counts.begin(), [num_domains](int n){
                        return n * num_domains;
                    });
                    std::vector<int> all_num_counts_displs(size); // 3/6
                    for (std::size_t r_idx = 1; r_idx < all_num_counts_displs.size(); ++r_idx) {
                        all_num_counts_displs[r_idx] = all_num_counts_displs[r_idx - 1] + all_num_counts[r_idx - 1];
                    }
                    std::vector<int> all_flat_send_counts(all_flat_recv_counts.size()); // 4/6 (output)
                    // all_num_counts // 5/6
                    // all_num_counts_displs // 6/6

                    // First all-to-all
                    comm.all_to_allv(all_flat_recv_counts, all_num_counts, all_num_counts_displs,
                                     all_flat_send_counts, all_num_counts, all_num_counts_displs);

                    // Setup second all-to-all (1-6): recv / send indices
                    std::vector<index_type> all_flat_recv_indices{}; // 1/6
                    for (const auto& ri_map : all_recv_indices) {
                        for (const auto& ri_pair : ri_map) {
                            all_flat_recv_indices.insert(all_flat_recv_indices.end(), ri_pair.second.begin(), ri_pair.second.end());
                        }
                    }
                    std::vector<int> all_num_recv_indices(size); // 2/6
                    for (std::size_t r_idx = 0; r_idx < all_recv_indices.size(); ++r_idx) {
                        for (const auto& ri_pair : all_recv_indices[r_idx]) {
                            all_num_recv_indices[r_idx] += ri_pair.second.size();
                        }
                    }
                    std::vector<int> all_num_recv_indices_displs(size); // 3/6
                    for (std::size_t r_idx = 1; r_idx < all_num_recv_indices_displs.size(); ++r_idx) {
                        all_num_recv_indices_displs[r_idx] = all_num_recv_indices_displs[r_idx - 1] + all_num_recv_indices[r_idx - 1];
                    }
                    std::size_t tot_send_counts = static_cast<std::size_t>(std::accumulate(all_flat_send_counts.begin(), all_flat_send_counts.end(), 0));
                    std::vector<index_type> all_flat_send_indices(tot_send_counts); // 4/6 (output)
                    std::vector<int> all_rank_send_counts(size); // 5/6
                    auto all_flat_send_counts_it = all_flat_send_counts.begin();
                    for (std::size_t r_idx = 0; r_idx < all_rank_send_counts.size(); ++r_idx) {
                        auto num_counts = all_num_domains[r_idx] * d_range.size();
                        all_rank_send_counts[r_idx] = std::accumulate(all_flat_send_counts_it, all_flat_send_counts_it + num_counts, 0); // TO DO: static cast of num_counts?
                        all_flat_send_counts_it += num_counts;
                    }
                    std::vector<int> all_rank_send_counts_displs(size); // 6/6
                    for (std::size_t r_idx = 1; r_idx < all_rank_send_counts_displs.size(); ++r_idx) {
                        all_rank_send_counts_displs[r_idx] = all_rank_send_counts_displs[r_idx - 1] + all_rank_send_counts[r_idx - 1];
                    }

                    // Second all-to-all
                    comm.all_to_allv(all_flat_recv_indices, all_num_recv_indices, all_num_recv_indices_displs,
                                     all_flat_send_indices, all_rank_send_counts, all_rank_send_counts_displs);

                    // Reconstruct multidimensional objects
                    all_send_counts_type all_send_counts(size);
                    all_flat_send_counts_it = all_flat_send_counts.begin();
                    for (std::size_t r_idx = 0; r_idx < all_send_counts.size(); ++r_idx) {
                        for (auto d_id : domain_ids) {
                            for (int other_d_idx = 0; other_d_idx < all_num_domains[r_idx]; ++other_d_idx) {
                                all_send_counts[r_idx][d_id].push_back(*all_flat_send_counts_it++);
                            }
                        }
                    }

                    // Setup send halos
                    auto all_flat_send_indices_it = all_flat_send_indices.begin();
                    for (std::size_t r_idx = 0; r_idx < all_send_counts.size(); ++r_idx) {
                        for (std::size_t d_idx = 0; d_idx < domain_ids.size(); ++ d_idx) {
                            auto d_id = domain_ids[d_idx];
                            auto levels = hgen(d_range[local_domain_ids_map.at(d_id)]).levels();
                            auto send_counts = all_send_counts[r_idx].at(d_id);
                            for (std::size_t other_d_idx = 0; other_d_idx < send_counts.size(); ++other_d_idx) {
                                if (send_counts[other_d_idx]) {
                                    auto other_id = all_domain_ids[r_idx][other_d_idx];
                                    int other_rank = static_cast<int>(r_idx);
                                    auto other_address = all_addresses[r_idx];
                                    int tag = (static_cast<int>(d_id) << 7) + static_cast<int>(other_id); // TO DO: maximum shift should not be hard-coded
                                    extended_domain_id_type id{other_id, other_rank, other_address, tag};
                                    iteration_space_type is{{all_flat_send_indices_it, all_flat_send_indices_it + send_counts[other_d_idx]}, levels}; // TO DO: static cast of send_counts[other_d_idx]?
                                    index_container_type ic{is};
                                    my_patterns[d_idx].send_halos().insert({id, ic});
                                    all_flat_send_indices_it += send_counts[other_d_idx]; // TO DO: static cast of send_counts[other_d_idx]?
                                }
                            }
                        }
                    }

                    return pattern_container<communicator_type, grid_type, domain_id_type>(std::move(my_patterns), m_max_tag);

                }

            };

        } // namespace detail

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_UNSTRUCTURED_PATTERN_HPP */
