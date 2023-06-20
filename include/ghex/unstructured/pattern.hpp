/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/context.hpp>
#include <ghex/mpi/communicator.hpp>
#include <ghex/device/cuda/unified_memory_allocator.hpp>
#include <ghex/pattern_container.hpp>
#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/grid.hpp>

#include <vector>
#include <set>
#include <map>
#include <numeric>
#include <algorithm>
#include <iterator>
#include <iosfwd>
#include <iostream>

namespace ghex
{
/** @brief unstructured pattern specialization
 *
 * This class provides access to the receive and send iteration spaces, determined by the halos,
 * and holds all connections to the neighbors.
 *
 * @tparam Index index type for domain (local indices) and iteration space
 * @tparam DomainId domain id type*/
template<typename Index, typename DomainId>
class pattern<unstructured::detail::grid<Index>, DomainId>
{
  public:
    // member types
    using index_type = Index;
    using domain_id_type = DomainId;
    using communicator_type = oomph::communicator;
    using rank_type = oomph::rank_type;
    using grid_type = unstructured::detail::grid<index_type>;
    using pattern_container_type = pattern_container<grid_type, domain_id_type>;

    friend class pattern_container<grid_type, domain_id_type>;

    /** @brief unstructured iteration space for accessing halo elements*/
    class iteration_space
    {
      public:
        using allocator_type = ghex::allocator::cuda::unified_memory_allocator<index_type>;
        using local_indices_type = std::vector<index_type, allocator_type>;

      private:
        local_indices_type m_local_indices;

      public:
        // ctors
        iteration_space() noexcept = default;

        iteration_space(local_indices_type local_indices)
        : m_local_indices{std::move(local_indices)}
        {
        }

        // member functions
        std::size_t               size() const noexcept { return m_local_indices.size(); }
        const local_indices_type& local_indices() const noexcept { return m_local_indices; }
        local_indices_type&       local_indices() noexcept { return m_local_indices; }

        void push_back(const index_type idx) { m_local_indices.push_back(idx); }

        // print
        /** @brief print */
        template<typename CharT, typename Traits>
        friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
            const iteration_space&                                                              is)
        {
            os << "size = " << is.size() << ";\n"
               << "local indices: [ ";
            for (auto idx : is.local_indices()) { os << idx << " "; }
            os << "]\n";
            return os;
        }
    };

    // TO DO: should be simplified, just one halo per local domain
    using index_container_type = std::vector<iteration_space>;

    /** @brief extended domain id, including rank, address and tag information*/
    struct extended_domain_id_type
    {
        // members
        domain_id_type id;
        int            mpi_rank;
        int            tag;

        // member functions
        /** @brief unique ordering given by address and tag*/
        bool operator<(const extended_domain_id_type& other) const noexcept
        {
            return mpi_rank < other.mpi_rank
                       ? true
                       : (mpi_rank == other.mpi_rank ? (tag < other.tag) : false);
        }

        /** @brief print*/
        template<typename CharT, typename Traits>
        friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
            const extended_domain_id_type& ext_id)
        {
            os << "{id = " << ext_id.id << ", rank = " << ext_id.mpi_rank
               << ", tag = " << ext_id.tag << "}\n";
            return os;
        }
    };

    // halo map type
    using map_type = std::map<extended_domain_id_type, index_container_type>;

    // static member functions
    /** @brief compute number of elements in an object of type index_container_type*/
    static std::size_t num_elements(const index_container_type& c) noexcept
    {
        std::size_t s{0};
        for (const auto& is : c) s += is.size();
        return s;
    }

  private:
    // members
    extended_domain_id_type m_id;
    map_type                m_send_map;
    map_type                m_recv_map;
    pattern_container_type* m_container;

  public:
    // ctors
    pattern(const extended_domain_id_type& id)
    : m_id{id}
    , m_send_map{}
    , m_recv_map{}
    , m_container{nullptr}
    {
    }
    pattern(const pattern&) = default;
    pattern(pattern&&) = default;

    // member functions
    domain_id_type                domain_id() const noexcept { return m_id.id; }
    extended_domain_id_type       extended_domain_id() const noexcept { return m_id; }
    map_type&                     send_halos() noexcept { return m_send_map; }
    const map_type&               send_halos() const noexcept { return m_send_map; }
    map_type&                     recv_halos() noexcept { return m_recv_map; }
    const map_type&               recv_halos() const noexcept { return m_recv_map; }
    const pattern_container_type& container() const noexcept { return *m_container; }

    /** @brief tie pattern to field
     * @tparam Field field type
     * @param field field instance
     * @return buffer_info object which holds pointers to the field and the pattern*/
    template<typename Field>
    buffer_info<pattern, typename Field::arch_type, Field> operator()(Field& field) const
    {
        return {*this, field, field.device_id()};
    }
};

namespace detail
{
/** @brief constructs the pattern with the help of all to all communications*/
template<typename Index>
struct make_pattern_impl<unstructured::detail::grid<Index>>
{
    static constexpr unsigned num_bits(unsigned n)
    {
        if (!n) return 1u;
        return 1u + num_bits(n >> 1);
    }

    /** @brief specialization used when no hints on neighbor domains are provided*/
    template<typename HaloGenerator, typename DomainRange>
    static auto apply(context& ctxt, HaloGenerator&& hgen, DomainRange&& d_range)
    {
        using grid_type = unstructured::detail::grid<Index>;
        using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
        using domain_id_type = typename domain_type::domain_id_type;
        using global_index_type = typename domain_type::global_index_type;
        using pattern_type = pattern<grid_type, domain_id_type>;
        using iteration_space_type = typename pattern_type::iteration_space;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
        using index_container_type = typename pattern_type::index_container_type;

        // get setup comm, and then this rank and size
        auto comm = mpi::communicator(ctxt);
        auto my_rank = comm.rank();
        auto size = comm.size();

        // setup patterns
        std::vector<pattern_type> my_patterns;
        for (const auto& d : d_range)
        {
            pattern_type p{{d.domain_id(), my_rank, 0}};
            my_patterns.push_back(p);
        }

        // get maximum domain id (used to create tags)
        auto my_max_domain_id = max_element(my_patterns.begin(), my_patterns.end(),
            [](const auto& lhs, const auto& rhs) {
                return lhs.domain_id() < rhs.domain_id();
            })->domain_id();
        auto all_max_domain_ids = comm.all_gather(my_max_domain_id).get();
        auto max_domain_id = *max_element(all_max_domain_ids.begin(), all_max_domain_ids.end());

        // gather halos from all local domains on all ranks.
        int  my_num_domains = d_range.size();
        auto all_num_domains = comm.all_gather(my_num_domains).get();

        // create tag scheme
        unsigned max_num_domains =
            *std::max_element(all_num_domains.begin(), all_num_domains.end());
        auto make_tag = [shift = num_bits(max_num_domains)](unsigned src_local_domain_id,
                            domain_id_type                           tgt_domain_id) -> int
        { return (src_local_domain_id << shift) | tgt_domain_id; };
        int max_tag = make_tag(max_num_domains, max_domain_id);

        // POD data for send halo exchange
        struct domain_data
        {
            domain_id_type id;
            std::size_t    halo_size;
        };
        std::vector<domain_data> my_domain_data;

        // single reduced halo with halo gids of all local domains
        std::vector<global_index_type> my_reduced_halos;

        // loop over domains and get halo and domain data
        for (const auto& d : d_range)
        {
            auto h = hgen(d);
            std::transform(h.local_indices().begin(), h.local_indices().end(),
                std::back_inserter(my_reduced_halos),
                [&d](auto lid) { return d.global_index(lid).value(); });
            my_domain_data.push_back(domain_data{d.domain_id(), h.size()});
        }

        // exchange domain data
        auto all_domain_data = comm.all_gather(my_domain_data, all_num_domains).get();

        // exchange reduced halo data
        std::vector<int> all_reduced_halos_sizes(size, 0);
        std::transform(all_domain_data.begin(), all_domain_data.end(),
            all_reduced_halos_sizes.begin(),
            [](const std::vector<domain_data>& d_vec)
            {
                return std::accumulate(d_vec.begin(), d_vec.end(), 0,
                    [](int i, const auto& d) -> int { return i + d.halo_size; });
            });
        auto all_reduced_halos = comm.all_gather(my_reduced_halos, all_reduced_halos_sizes).get();

        // POD data for recv halo exchange
        struct recv_halo_data
        {
            domain_id_type id;
            domain_id_type other_id;
            int            recv_rank;
            int            tag;
            std::size_t    is_size;
        };
        std::vector<recv_halo_data> my_recv_halo_data;

        // loop over all ranks and create send halos
        for (int other_rank = 0; other_rank < size; ++other_rank)
        {
            std::size_t offset = 0u;
            const auto& other_reduced_halos = all_reduced_halos[other_rank];
            unsigned    other_domain_count = 0u;
            for (const domain_data& other_d : all_domain_data[other_rank])
            {
                auto     first = other_reduced_halos.begin() + offset;
                auto     last = other_reduced_halos.begin() + offset + other_d.halo_size;
                unsigned my_domain_count = 0u;
                for (const auto& d : d_range)
                {
                    int                     tag = make_tag(my_domain_count, other_d.id);
                    extended_domain_id_type id{other_d.id, other_rank, tag};
                    iteration_space_type    is;
                    for (auto it = first; it != last; ++it)
                    {
                        auto gid = *it;
                        if (d.is_inner(gid)) { is.push_back(d.inner_local_index(gid).value()); }
                    }
                    if (is.size())
                    {
                        my_recv_halo_data.push_back(
                            recv_halo_data{d.domain_id(), other_d.id, other_rank, tag, is.size()});
                        my_patterns[my_domain_count].send_halos().insert(
                            std::make_pair(id, index_container_type{std::move(is)}));
                    }
                    ++my_domain_count;
                }
                offset += other_d.halo_size;
                ++other_domain_count;
            }
        }

        // exchange sizes of all receive halos
        auto all_recv_halo_data_sizes = comm.all_gather((int)my_recv_halo_data.size()).get();
        auto all_recv_halo_data =
            comm.all_gather(my_recv_halo_data, all_recv_halo_data_sizes).get();

        // data structues for non-blocking exchanges
        std::vector<mpi::future<void>>              futures;
        std::vector<std::vector<global_index_type>> send_indices;

        // loop over send halo indices, transform them to global indices and send them to their
        // neighbors (non-blocking)
        unsigned domain_counter = 0;
        for (auto& p : my_patterns)
        {
            const auto& d = d_range[domain_counter];
            for (auto& x : p.send_halos())
            {
                std::vector<global_index_type> tmp;
                tmp.reserve(x.second.front().size());
                std::transform(x.second.front().local_indices().begin(),
                    x.second.front().local_indices().end(), std::back_inserter(tmp),
                    [&d](auto lid) { return d.global_index(lid).value(); });
                futures.push_back(
                    comm.isend(x.first.mpi_rank, x.first.tag, tmp.data(), tmp.size()));
                send_indices.push_back(std::move(tmp));
            }
            ++domain_counter;
        }

        // create receive halos: wait for global indices to arrive from neighbors and then transform
        // them into local indices again
        std::vector<global_index_type> tmp;
        for (int other_rank = 0; other_rank < size; ++other_rank)
        {
            for (const auto r : all_recv_halo_data[other_rank])
            {
                if (r.recv_rank == my_rank)
                {
                    extended_domain_id_type id{r.id, other_rank, r.tag};
                    domain_counter = 0;
                    for (auto& p : my_patterns)
                    {
                        if (p.domain_id() == r.other_id)
                        {
                            auto&                d = d_range[domain_counter];
                            iteration_space_type is;
                            tmp.resize(r.is_size);
                            comm.recv(other_rank, r.tag, tmp.data(), r.is_size);
                            auto lids = d.make_outer_lids(tmp);
                            assert(lids.size() == tmp.size());
                            is.local_indices().assign(lids.begin(), lids.end());
                            p.recv_halos().insert(
                                std::make_pair(id, index_container_type{std::move(is)}));
                        }
                        ++domain_counter;
                    }
                }
            }
        }

        // wait for all communication to finish
        for (auto& f : futures) f.wait();

        return pattern_container<grid_type, domain_id_type>(ctxt, std::move(my_patterns), max_tag);
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
    template<typename HaloGenerator, typename RecvDomainIdsGen, typename DomainRange>
    static auto apply(context& ctxt, HaloGenerator&& hgen, RecvDomainIdsGen&& recv_domain_ids_gen,
        DomainRange&& d_range)
    {
        // typedefs
        using grid_type = unstructured::detail::grid<Index>;
        using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
        using domain_id_type = typename domain_type::domain_id_type;
        using pattern_type = pattern<grid_type, domain_id_type>;
        using index_type = typename pattern_type::index_type;
        using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
        using iteration_space_type = typename pattern_type::iteration_space;
        using index_container_type = typename pattern_type::index_container_type;

        using all_recv_counts_type = std::vector<std::map<domain_id_type, std::vector<int>>>;
        using all_send_counts_type = std::vector<std::map<domain_id_type, std::vector<int>>>;
        using all_recv_indices_type =
            std::vector<std::map<domain_id_type, std::vector<index_type>>>;

        // get setup comm, and then this rank and size
        auto comm = mpi::communicator(ctxt);
        auto my_rank = comm.rank();
        auto size = comm.size();

        // gather domain ids all ranks
        // number of local domains (int, since has to be used as elem counter)
        int num_domains = d_range.size();
        // numbers of local domains on all ranks
        auto all_num_domains = comm.all_gather(num_domains).get();
        // map between local domain ids and indices in d_range
        std::map<domain_id_type, std::size_t> local_domain_ids_map{};
        for (std::size_t idx = 0; idx < d_range.size(); ++idx)
        {
            local_domain_ids_map.insert({d_range[idx].domain_id(), idx});
        }
        // domain id for each local domain (ordered)
        std::vector<domain_id_type> domain_ids(d_range.size());
        std::transform(local_domain_ids_map.begin(), local_domain_ids_map.end(), domain_ids.begin(),
            [](auto p) { return p.first; });
        // domain id for each local domain on all ranks
        auto all_domain_ids = comm.all_gather(domain_ids, all_num_domains).get();

        // set max tag
        auto max_domain_id = comm.max_element(domain_ids);
        // max tag TO DO: maximum shift should not be hard-coded. TO DO: should add 1?
        int m_max_tag = (max_domain_id << 7) + max_domain_id;

        // helpers
        // helper domain_id to rank map (will be filled only with needed domain ids)
        std::map<domain_id_type, int> domain_id_to_rank;

        // recv side setup
        all_recv_counts_type all_recv_counts(size);
        for (auto other_rank = 0; other_rank < size; ++other_rank)
        {
            for (auto id : all_domain_ids[other_rank])
            {
                all_recv_counts[other_rank].insert({id, std::vector<int>(d_range.size(), 0)});
            }
        }
        all_recv_indices_type all_recv_indices(size);

        // setup patterns, with only recv halos for now
        std::vector<pattern_type> my_patterns;
        std::size_t               d_reidx{0};
        // sorted by domain id
        for (const auto& d_id_idx : local_domain_ids_map)
        {
            const auto& d = d_range[d_id_idx.second];
            // construct pattern
            pattern_type p{{d.domain_id(), my_rank, 0}};
            // local helper for filling iteration spaces
            std::map<domain_id_type, iteration_space_type> tmp_is_map{};
            auto                                           h = hgen(d);
            auto                                           r_ids = recv_domain_ids_gen(d);
            for (std::size_t h_idx = 0; h_idx < h.size(); ++h_idx)
            {
                domain_id_to_rank.insert({r_ids.domain_ids()[h_idx], r_ids.ranks()[h_idx]});
                tmp_is_map[r_ids.domain_ids()[h_idx]].push_back(h.local_indices()[h_idx]);
                all_recv_counts[r_ids.ranks()[h_idx]].at(r_ids.domain_ids()[h_idx])[d_reidx] += 1;
                all_recv_indices[r_ids.ranks()[h_idx]][r_ids.domain_ids()[h_idx]].push_back(
                    r_ids.remote_indices()[h_idx]);
            }
            for (const auto& d_id_is : tmp_is_map)
            {
                auto other_id = d_id_is.first;
                auto other_rank = domain_id_to_rank.at(other_id);
                // TO DO: maximum shift should not be hard-coded
                int tag = (static_cast<unsigned int>(other_id) << 7) + d.domain_id();
                extended_domain_id_type id{other_id, other_rank, tag};
                iteration_space_type    is{d_id_is.second.local_indices()};
                index_container_type    ic{is};
                p.recv_halos().insert({id, ic});
            }
            my_patterns.push_back(p);
            ++d_reidx;
        }

        // Setup first all-to-all (1-6): recv / send counts
        std::vector<int> all_flat_recv_counts{}; // 1/6
        for (const auto& rc_map : all_recv_counts)
        {
            for (const auto& rc_pair : rc_map)
            {
                all_flat_recv_counts.insert(all_flat_recv_counts.end(), rc_pair.second.begin(),
                    rc_pair.second.end());
            }
        }
        // same for send and recv
        std::vector<int> all_num_counts(size); // 2/6
        std::transform(all_num_domains.begin(), all_num_domains.end(), all_num_counts.begin(),
            [num_domains](int n) { return n * num_domains; });
        std::vector<int> all_num_counts_displs(size); // 3/6
        for (std::size_t r_idx = 1; r_idx < all_num_counts_displs.size(); ++r_idx)
        {
            all_num_counts_displs[r_idx] =
                all_num_counts_displs[r_idx - 1] + all_num_counts[r_idx - 1];
        }
        std::vector<int> all_flat_send_counts(all_flat_recv_counts.size()); // 4/6 (output)
        // all_num_counts // 5/6
        // all_num_counts_displs // 6/6

        // First all-to-all
        comm.all_to_allv(all_flat_recv_counts, all_num_counts, all_num_counts_displs,
            all_flat_send_counts, all_num_counts, all_num_counts_displs);

        // Setup second all-to-all (1-6): recv / send indices
        std::vector<index_type> all_flat_recv_indices{}; // 1/6
        for (const auto& ri_map : all_recv_indices)
        {
            for (const auto& ri_pair : ri_map)
            {
                all_flat_recv_indices.insert(all_flat_recv_indices.end(), ri_pair.second.begin(),
                    ri_pair.second.end());
            }
        }
        std::vector<int> all_num_recv_indices(size); // 2/6
        for (std::size_t r_idx = 0; r_idx < all_recv_indices.size(); ++r_idx)
        {
            for (const auto& ri_pair : all_recv_indices[r_idx])
            {
                all_num_recv_indices[r_idx] += ri_pair.second.size();
            }
        }
        std::vector<int> all_num_recv_indices_displs(size); // 3/6
        for (std::size_t r_idx = 1; r_idx < all_num_recv_indices_displs.size(); ++r_idx)
        {
            all_num_recv_indices_displs[r_idx] =
                all_num_recv_indices_displs[r_idx - 1] + all_num_recv_indices[r_idx - 1];
        }
        std::size_t tot_send_counts =
            std::accumulate(all_flat_send_counts.begin(), all_flat_send_counts.end(), 0);
        std::vector<index_type> all_flat_send_indices(tot_send_counts); // 4/6 (output)
        std::vector<int>        all_rank_send_counts(size);             // 5/6
        auto                    all_flat_send_counts_it = all_flat_send_counts.begin();
        for (std::size_t r_idx = 0; r_idx < all_rank_send_counts.size(); ++r_idx)
        {
            auto num_counts = all_num_domains[r_idx] * d_range.size();
            all_rank_send_counts[r_idx] =
                std::accumulate(all_flat_send_counts_it, all_flat_send_counts_it + num_counts, 0);
            all_flat_send_counts_it += num_counts;
        }
        std::vector<int> all_rank_send_counts_displs(size); // 6/6
        for (std::size_t r_idx = 1; r_idx < all_rank_send_counts_displs.size(); ++r_idx)
        {
            all_rank_send_counts_displs[r_idx] =
                all_rank_send_counts_displs[r_idx - 1] + all_rank_send_counts[r_idx - 1];
        }

        // Second all-to-all
        comm.all_to_allv(all_flat_recv_indices, all_num_recv_indices, all_num_recv_indices_displs,
            all_flat_send_indices, all_rank_send_counts, all_rank_send_counts_displs);

        // Reconstruct multidimensional objects
        all_send_counts_type all_send_counts(size);
        all_flat_send_counts_it = all_flat_send_counts.begin();
        for (std::size_t r_idx = 0; r_idx < all_send_counts.size(); ++r_idx)
        {
            for (auto d_id : domain_ids)
            {
                for (int other_d_idx = 0; other_d_idx < all_num_domains[r_idx]; ++other_d_idx)
                {
                    all_send_counts[r_idx][d_id].push_back(*all_flat_send_counts_it++);
                }
            }
        }

        // Setup send halos
        auto all_flat_send_indices_it = all_flat_send_indices.begin();
        for (std::size_t r_idx = 0; r_idx < all_send_counts.size(); ++r_idx)
        {
            for (std::size_t d_idx = 0; d_idx < domain_ids.size(); ++d_idx)
            {
                auto d_id = domain_ids[d_idx];
                auto send_counts = all_send_counts[r_idx].at(d_id);
                for (std::size_t other_d_idx = 0; other_d_idx < send_counts.size(); ++other_d_idx)
                {
                    if (send_counts[other_d_idx])
                    {
                        auto other_id = all_domain_ids[r_idx][other_d_idx];
                        int  other_rank = r_idx;
                        // TO DO: maximum shift should not be hard-coded
                        int tag = (static_cast<unsigned int>(d_id) << 7) + other_id;
                        extended_domain_id_type id{other_id, other_rank, tag};
                        iteration_space_type    is{{all_flat_send_indices_it,
                               all_flat_send_indices_it + send_counts[other_d_idx]}};
                        index_container_type    ic{is};
                        my_patterns[d_idx].send_halos().insert({id, ic});
                        all_flat_send_indices_it += send_counts[other_d_idx];
                    }
                }
            }
        }

        return pattern_container<grid_type, domain_id_type>(ctxt, std::move(my_patterns),
            m_max_tag);
    }
};

} // namespace detail

} // namespace ghex
