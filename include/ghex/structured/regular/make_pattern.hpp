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

#include <ghex/structured/pattern.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <array>
#include <vector>

namespace ghex
{
namespace structured
{
namespace regular
{
//        template<typename CoordinateArrayType>
//        struct make_pattern_impl<::gridtools::ghex::structured::detail::grid<CoordinateArrayType>>
//            template<typename Transport, typename HaloGenerator, typename DomainRange>
//            static auto apply(tl::context<Transport>& context, HaloGenerator&& hgen, DomainRange&& d_range)

namespace detail
{
template<typename DomainRange>
struct domain_traits
{
    using domain_type = typename std::remove_reference_t<DomainRange>::value_type;
    using coord = typename domain_type::coordinate_type;
    using grid_type_impl = structured::grid;
    using grid_type = typename grid_type_impl::template type<domain_type>;
    using coordinate_type = typename grid_type::coordinate_type;
    using dimension = typename grid_type::dimension;
    using halo_type = std::array<int, dimension::value * 2>;
    using periodic = std::array<bool, dimension::value>;
    using domain_id_type = typename domain_type::domain_id_type;
    using halo_gen_type =
        regular::halo_generator<domain_id_type, std::integral_constant<int, dimension::value>>;
};
} // namespace detail

template<typename DomainRange, typename DomainLookUp>
auto
make_staged_pattern(ghex::context& ctxt, DomainRange&& d_range, DomainLookUp&& d_lu,
    typename detail::domain_traits<DomainRange>::coord const&     g_first,
    typename detail::domain_traits<DomainRange>::coord const&     g_last,
    typename detail::domain_traits<DomainRange>::halo_type const& halos,
    typename detail::domain_traits<DomainRange>::periodic const&  period)
{
    //using communicator_type = oomph::communicator;
    using pattern_type = ghex::pattern<
        //communicator_type,
        //typename detail::domain_traits<DomainRange>::coord,
        typename detail::domain_traits<DomainRange>::grid_type,
        typename detail::domain_traits<DomainRange>::domain_id_type>;
    //using pattern_container_type = typename pattern_type::pattern_container_type;
    using pattern_container_type = ghex::pattern_container<
        //communicator_type,
        typename detail::domain_traits<DomainRange>::grid_type,
        typename detail::domain_traits<DomainRange>::domain_id_type>;
    using dimension = typename detail::domain_traits<DomainRange>::dimension;
    using iteration_space = typename pattern_type::iteration_space;
    using iteration_space_pair = typename pattern_type::iteration_space_pair;
    using coordinate_type = typename pattern_type::coordinate_type;
    using domain_id_type = typename pattern_type::domain_id_type;
    using extended_domain_id_type = typename pattern_type::extended_domain_id_type;
    //using h_gen_type = typename detail::domain_traits<DomainRange>::halo_gen_type;

    auto comm = ghex::mpi::communicator(ctxt); //tl::mpi::setup_communicator(context.mpi_comm());
    auto my_rank = comm.rank();

    std::vector<iteration_space_pair>    my_domain_extents;
    std::vector<extended_domain_id_type> my_domain_ids;

    std::array<std::vector<pattern_type>, dimension::value>               my_patterns;
    std::array<std::unique_ptr<pattern_container_type>, dimension::value> my_pattern_conts;
    //std::array<std::vector<std::vector<iteration_space_pair>>, dimension::value> my_recv_halos;
    //std::array<std::vector<std::vector<extended_domain_id_type>>, dimension::value> my_recv_halo_ids;
    //std::array<std::vector<std::vector<iteration_space_pair>>, dimension::value> my_send_halos;
    //std::array<std::vector<std::vector<extended_domain_id_type>>, dimension::value> my_send_halo_ids;

    for (const auto& d : d_range)
    {
        my_domain_ids.push_back(extended_domain_id_type{d.domain_id(), my_rank, 0});
        auto is_pair = iteration_space_pair{
            iteration_space{coordinate_type{d.first()} - coordinate_type{d.first()},
                coordinate_type{d.last()} - coordinate_type{d.first()}},
            iteration_space{coordinate_type{d.first()}, coordinate_type{d.last()}}};
        my_domain_extents.push_back(is_pair);

        for (unsigned int i = 0; i < dimension::value; ++i)
        {
            my_patterns[i].emplace_back(my_domain_extents.back(), my_domain_ids.back());
            my_patterns[i].back().global_first() = g_first;
            my_patterns[i].back().global_last() = g_last;
            //std::cout << "i=" << i << std::endl;

            //my_recv_halos[i].resize(my_recv_halos[i].size()+1);
            //my_recv_halo_ids[i].resize(my_recv_halo_ids[i].size()+1);
            //my_send_halos[i].resize(my_send_halos[i].size()+1);
            //my_send_halo_idss[i].resize(my_send_halo_ids[i].size()+1);
            const bool has_left =
                halos[2 * i] > 0 &&
                (period[i] || ((is_pair.global().first()[i] - halos[2 * i]) >= g_first[i]));
            const bool has_right =
                halos[2 * i + 1] > 0 &&
                (period[i] || ((is_pair.global().last()[i] + halos[2 * i + 1]) <= g_last[i]));
            std::array<int, dimension::value> neighbor_offset;
            neighbor_offset.fill(0);
            if (has_left)
            {
                neighbor_offset[i] = -1;
                auto neighbor = d_lu(d.domain_id(), neighbor_offset);
                //std::cout << "has_left: " << neighbor.rank() << ", " << neighbor.id() << std::endl;
                auto neighbor_eid = extended_domain_id_type{neighbor.id(), neighbor.rank(), 0};
                auto recv_left = is_pair;
                recv_left.local().last()[i] = recv_left.local().first()[i] - 1;
                recv_left.local().first()[i] -= halos[2 * i];
                recv_left.global().last()[i] = recv_left.global().first()[i] - 1;
                recv_left.global().first()[i] -= halos[2 * i];
                //my_recv_halos[i].back().push_back(recv_left);
                //my_recv_halo_ids[i].back().push_back(neighbor_eid);
                my_patterns[i].back().recv_halos()[neighbor_eid].push_back(recv_left);
            }
            if (has_right)
            {
                neighbor_offset[i] = 1;
                auto neighbor = d_lu(d.domain_id(), neighbor_offset);
                auto neighbor_eid = extended_domain_id_type{neighbor.id(), neighbor.rank(), 0};
                auto send_right = is_pair;
                send_right.local().first()[i] = send_right.local().last()[i] + 1 - halos[2 * i];
                send_right.global().first()[i] = send_right.global().last()[i] + 1 - halos[2 * i];
                //my_send_halos[i].back().push_back(send_right);
                //my_send_halo_ids[i].back().push_back(neighbor_eid);
                my_patterns[i].back().send_halos()[neighbor_eid].push_back(send_right);
            }
            if (has_right)
            {
                neighbor_offset[i] = 1;
                auto neighbor = d_lu(d.domain_id(), neighbor_offset);
                auto neighbor_eid = extended_domain_id_type{neighbor.id(), neighbor.rank(), 0};
                auto recv_right = is_pair;
                recv_right.local().first()[i] = recv_right.local().last()[i] + 1;
                recv_right.global().first()[i] = recv_right.global().last()[i] + 1;
                recv_right.local().last()[i] += halos[2 * i + 1];
                recv_right.global().last()[i] += halos[2 * i + 1];
                //my_recv_halos[i].back().push_back(recv_right);
                //my_recv_halo_ids[i].back().push_back(neighbor_eid);
                my_patterns[i].back().recv_halos()[neighbor_eid].push_back(recv_right);
            }
            if (has_left)
            {
                neighbor_offset[i] = -1;
                auto neighbor = d_lu(d.domain_id(), neighbor_offset);
                auto neighbor_eid = extended_domain_id_type{neighbor.id(), neighbor.rank(), 0};
                auto send_left = is_pair;
                send_left.local().last()[i] = send_left.local().first()[i] - 1 + halos[2 * i + 1];
                send_left.global().last()[i] = send_left.global().first()[i] - 1 + halos[2 * i + 1];
                //my_send_halos[i].back().push_back(send_left);
                //my_send_halo_ids[i].back().push_back(neighbor_eid);
                my_patterns[i].back().send_halos()[neighbor_eid].push_back(send_left);
            }
            if (has_left)
            {
                is_pair.local().first()[i] -= halos[2 * i];
                is_pair.global().first()[i] -= halos[2 * i];
            }
            if (has_right)
            {
                is_pair.local().last()[i] += halos[2 * i + 1];
                is_pair.global().last()[i] += halos[2 * i + 1];
            }
        }
    }

    struct tag_item
    {
        domain_id_type source_id;
        domain_id_type dest_id;
        int            tag;
    };

    for (unsigned int i = 0; i < dimension::value; ++i)
    {
        int                                  m_max_tag = 0;
        std::map<int, std::vector<tag_item>> tag_map;
        std::map<int, std::vector<tag_item>> send_tag_map;
        for (auto& p : my_patterns[i])
        {
            // loop over all halos
            for (auto& id_is_pair : p.recv_halos())
            {
                // get rank from extended domain id
                const int rank = id_is_pair.first.mpi_rank;
                auto      it = tag_map.find(rank);
                if (it == tag_map.end())
                {
                    // if neighbor rank is not yet in tag_map: insert it with maximum tag = 0
                    tag_map[rank].push_back(tag_item{id_is_pair.first.id, p.domain_id(), 0});
                    // set tag in the receive halo data structure to 0
                    const_cast<extended_domain_id_type&>(id_is_pair.first).tag = 0;
                }
                else
                {
                    // if neighbor rank is already present: increase the tag by 1
                    auto last_tag = it->second.back().tag;
                    //it->second.emplace_back(rank, my_rank, last_tag+1);
                    it->second.push_back(
                        tag_item{id_is_pair.first.id, p.domain_id(), last_tag + 1});
                    m_max_tag = std::max(last_tag + 1, m_max_tag);
                    // set tag in the receive halo data structure
                    const_cast<extended_domain_id_type&>(id_is_pair.first).tag = last_tag + 1;
                }
            }
            for (auto& id_is_pair : p.send_halos())
            {
                const int rank = id_is_pair.first.mpi_rank;
                auto&     vec = send_tag_map[rank];
                vec.resize(vec.size() + 1);
            }
        }
        std::vector<MPI_Request> reqs(send_tag_map.size());
        int                      j = 0;
        for (auto& ti_vec_p : send_tag_map)
            MPI_Irecv(ti_vec_p.second.data(), ti_vec_p.second.size() * sizeof(tag_item), MPI_BYTE,
                ti_vec_p.first, 0, comm, &reqs[j++]);
        for (auto& ti_vec_p : tag_map)
            MPI_Send(ti_vec_p.second.data(), ti_vec_p.second.size() * sizeof(tag_item), MPI_BYTE,
                ti_vec_p.first, 0, comm);
        MPI_Waitall(reqs.size(), reqs.data(), MPI_STATUSES_IGNORE);

        for (auto& p : my_patterns[i])
        {
            // loop over all halos
            for (auto& id_is_pair : p.send_halos())
            {
                auto&          ti_vec = send_tag_map[id_is_pair.first.mpi_rank];
                domain_id_type source_id = p.domain_id();
                domain_id_type dest_id = id_is_pair.first.id;
                auto           tag =
                    std::find_if(ti_vec.begin(), ti_vec.end(), [source_id, dest_id](const auto& x) {
                        return x.source_id == source_id && x.dest_id == dest_id;
                    })->tag;
                const_cast<extended_domain_id_type&>(id_is_pair.first).tag = tag;
            }
        }
        my_pattern_conts[i] =
            std::make_unique<pattern_container_type>(ctxt, std::move(my_patterns[i]), m_max_tag);
    }
    return my_pattern_conts;
}

//        std::array<int,dimension::value> neighbor_offset;
//        neighbor_offset.fill(0);
//        for (unsigned int i=0; i<dimension::value; ++i)
//        {
//            // left halo
//            {
//                std::array<int,dimension::value*2> _halos;
//                _halos.fill(0);
//                _halos[i*2] = halos[i*2];
//                h_gen_type hgen(g_first, g_last, _halos, period, false);
//            }
//    //typename detail::domain_traits<DomainRange>::halo_gen_type hgen(g_first, g_last, halos, period, false);
//            auto offset = neighbor_offset;
//            offset[i] = -1;
//            auto left = d_lu.neighbor(d.domain_id(), offset);
//            offset[i] = 1;
//            auto right = d_lu.neighbor(d.domain_id(), offset);
//
//            if (left)
//            {
//                left.rank()
//                left.id()
//            }
//        }
//
//        auto recv_halos = hgen(d);
//        for (unsigned int i=0; i<dimension::value; ++i)
//        {
//            my_patterns[i].emplace_back(my_domain_extents.back(), my_domain_ids.back());
//            auto recv_halo_lo = recv_halos[i*2];
//            auto recv_halo_hi = recv_halos[i*2+1];
//
//            for (unsigned int j=0; j<i; ++j)
//            {
//                recv_halo_lo.local().first()[j] -= halos[2*j];
//                recv_halo_lo.local().last()[j]  += halos[2*j+1];
//                recv_halo_hi.local().first()[j] -= halos[2*j];
//                recv_halo_hi.local().last()[j]  += halos[2*j+1];
//            }
//        }
//
//        auto is_pair_lo = is_pair;
//        auto is_pair_hi = is_pair;
//        for (unsigned int i = 0; i<dimension::value; ++i)
//        {
//            is_pair_lo.local().last()[i]   = is_pair_lo.local().first()[i];
//            is_pair_lo.global().last()[i]  = is_pair_lo.global().first()[i];
//            is_pair_hi.local().first()[i]  = is_pair_hi.local().last()[i];
//            is_pair_hi.global().first()[i] = is_pair_hi.global().last()[i];
//        }
//        for (unsigned int i = 0; i<dimension::value; ++i)
//        {
//            my_patterns[i].emplace_back(my_domain_extents.back(), my_domain_ids.back());
//            is_pair_lo.local().first()[i] -= halos[i*2];
//        }
//                    // make space for more halos
//                    my_generated_recv_halos.resize(my_generated_recv_halos.size()+1);
//                    // generate recv halos: invoke halo generator
//                    auto recv_halos = hgen(d);
//                    // convert the recv halos to internal format
//                    for (const auto& h : recv_halos)
//                    {
//                        iteration_space_pair is{
//                            iteration_space{coordinate_type{h.local().first()},coordinate_type{h.local().last()}},
//                            iteration_space{coordinate_type{h.global().first()},coordinate_type{h.global().last()}}};
//                        // check that invariant is fullfilled (halos are not empty)
//                        if (is.local().first() <= is.local().last())
//                        {
//                            my_generated_recv_halos.back().push_back( is );
//                        }
//                    }
//                }
//
//}

} // namespace regular
} // namespace structured
} // namespace ghex
