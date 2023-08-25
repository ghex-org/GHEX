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

        // This alorithm uses 3 all-to-all communications and several (non-blocking) send and
        // receive communications (one for each halo/neighbor) to determine the halos. The
        // collectives are implemented by using a series of point-to-point communications in order
        // to avoid storing the complete set of all halos. This may be somewhat slower than native
        // MPI all-to-all operations but easily scales to large numbers of domains/halos.

        // get setup comm and this rank
        auto comm = mpi::communicator(ctxt);
        auto my_rank = comm.rank();

        // create patterns and store domain ids
        std::vector<pattern_type>   my_patterns;
        std::vector<domain_id_type> my_domain_ids;
        for (const auto& d : d_range)
        {
            my_patterns.emplace_back(pattern_type{{d.domain_id(), my_rank, 0}});
            my_domain_ids.push_back(d.domain_id());
        }

        // create tags by collecting domain ids from all domains
        auto max_domain_id = *std::max_element(my_domain_ids.begin(), my_domain_ids.end());
        auto max_num_domains = my_domain_ids.size();
        comm.distributed_for_each(
            [&](int, auto other_domain_ids)
            {
                max_domain_id =
                    std::max(*std::max_element(other_domain_ids.begin(), other_domain_ids.end()),
                        max_domain_id);
                max_num_domains = std::max(other_domain_ids.size(), max_num_domains);
            },
            my_domain_ids);
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

        // loop over local domains and get reduced halo and domain data
        for (const auto& d : d_range)
        {
            auto h = hgen(d);
            std::transform(h.local_indices().begin(), h.local_indices().end(),
                std::back_inserter(my_reduced_halos),
                [&d](auto lid) { return d.global_index(lid).value(); });
            my_domain_data.push_back(domain_data{d.domain_id(), h.size()});
        }

        // collect references to local domains and patterns for easier access
        std::vector<std::tuple<std::size_t, const domain_type&, pattern_type&>> pattern_tuples;
        {
            std::size_t i = 0u;
            for (const auto& d : d_range)
            {
                pattern_tuples.emplace_back(i, d, my_patterns[i]);
                ++i;
            }
        }

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

        // data structues for non-blocking exchanges
        std::vector<mpi::future<void>>              futures;
        std::vector<std::vector<global_index_type>> send_indices;

        // loop over each rank's domain data vectors and reduced halo data vectors in order to
        // create send halos
        comm.distributed_for_each(
            [&](int rank, auto other_domain_data, auto other_reduced_halos)
            {
                std::size_t offset = 0u; // offset into reduced halo data
                // loop over neighbor rank's domain data
                for (const domain_data& other_d : other_domain_data)
                {
                    // cut out appropriate halo indices from reduced halo data
                    auto first = other_reduced_halos.begin() + offset;
                    auto last = other_reduced_halos.begin() + offset + other_d.halo_size;
                    // loop over this rank's patterns
                    for (auto& [i, d, p] : pattern_tuples)
                    {
                        // cerate tag and id for potential communication connection
                        int                     tag = make_tag(i, other_d.id);
                        extended_domain_id_type id{other_d.id, rank, tag};
                        // create iteration space consisting of local indices iff any of the
                        // neighbors reduced halo gids are among inner indices of current domain d
                        iteration_space_type is;
                        for (auto it = first; it != last; ++it)
                        {
                            auto gid = *it;
                            if (d.is_inner(gid)) { is.push_back(d.inner_local_index(gid).value()); }
                        }
                        if (!is.size()) continue;
                        // fill recv halo data
                        my_recv_halo_data.push_back(
                            recv_halo_data{d.domain_id(), other_d.id, rank, tag, is.size()});
                        // start a non-blocking send to neighbor rank with the gid's that were
                        // determined above by checking whether they were among inner indices of d
                        std::vector<global_index_type> tmp;
                        tmp.reserve(is.size());
                        std::transform(is.local_indices().begin(), is.local_indices().end(),
                            std::back_inserter(tmp),
                            // use init-capture of address since clang doesn't allow lambda captures
                            // of structured bindings, (see https://stackoverflow.com/a/46115028)
                            [d = &d](auto lid) { return d->global_index(lid).value(); });
                        futures.push_back(comm.isend(rank, tag + 1, tmp.data(), tmp.size()));
                        send_indices.push_back(std::move(tmp));
                        // update the pattern's send halos
                        p.send_halos().insert(
                            std::make_pair(id, index_container_type{std::move(is)}));
                    }
                    offset += other_d.halo_size;
                }
            },
            my_domain_data, my_reduced_halos);

        // temporary data structure to store received global indices
        std::vector<global_index_type> tmp;

        // loop over each rank's recv halo data vectors which were filled above in order to create
        // recv halos
        comm.distributed_for_each(
            [&](int rank, auto other_recv_halo_data)
            {
                for (const auto r : other_recv_halo_data)
                {
                    // consider each recv halo data object only if its recv_rank equals this rank
                    if (r.recv_rank != my_rank) continue;
                    // create id and find corresponding pattern
                    extended_domain_id_type id{r.id, rank, r.tag};
                    for (auto& [i, d, p] : pattern_tuples)
                    {
                        if (d.domain_id() == r.other_id)
                        {
                            // receive subset remotely determined set of gids sent above
                            tmp.resize(r.is_size);
                            comm.recv(rank, r.tag + 1, tmp.data(), r.is_size);
                            // transform the gids into lids and update the pattern's receive halo
                            auto lids = d.make_outer_lids(tmp);
                            assert(lids.size() == tmp.size());
                            iteration_space_type is;
                            is.local_indices().assign(lids.begin(), lids.end());
                            p.recv_halos().insert(
                                std::make_pair(id, index_container_type{std::move(is)}));
                            break;
                        }
                    }
                }
            },
            my_recv_halo_data);

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
