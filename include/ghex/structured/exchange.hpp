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
#ifndef INCLUDED_GHEX_STRUCTURED_EXCHANGE_HPP
#define INCLUDED_GHEX_STRUCTURED_EXCHANGE_HPP

#include <vector>
#include <tuple>
#include <boost/mp11.hpp>
#include "../communication_object_2.hpp"
#include "../transport_layer/ri/range_factory.hpp"
#include "./pattern.hpp"

#include <iostream>

namespace gridtools {

namespace ghex {

namespace structured {

template<template <typename> class RangeGen, typename Communicator, typename Coordinate, typename DomainId, typename... Fields>
class bulk_communication_object
{
public: // member types
    using domain_id_type = DomainId;
    using grid_type = detail::grid<Coordinate>;
    using communicator_type = Communicator;
    using co_type = communication_object<communicator_type,grid_type,domain_id_type>;
    using co_handle = typename co_type::handle_type;
    using transport = typename communicator_type::tag_type;
    using pattern_container_type = pattern_container<communicator_type,grid_type,domain_id_type>;
    using pattern_type = typename pattern_container_type::value_type;
    using p_map_type = typename pattern_type::map_type;
    using iteration_space_pair = typename pattern_type::iteration_space_pair;

private: // member types
    template<typename A>
    struct select_arch_q
    {
        template<typename Field>
        using fn = typename Field::template rebind_arch<A>;
    };

    template<typename Field>
    using select_range = typename RangeGen<Field>::range_type;
    
    using field_types = boost::mp11::mp_list<Fields...>;
    using cpu_fields = boost::mp11::mp_transform_q<select_arch_q<cpu>,field_types>;
#ifdef GHEX_USE_GPU
    using gpu_fields = boost::mp11::mp_transform_q<select_arch_q<gpu>,field_types>;
    using all_fields = boost::mp11::mp_unique<boost::mp11::mp_append<cpu_fields,gpu_fields>>;
#else
    using all_fields = boost::mp11::mp_unique<cpu_fields>;
#endif
    using all_ranges = boost::mp11::mp_transform<select_range,all_fields>;
    using range_factory = tl::ri::range_factory<all_ranges>;

    template<typename Field>
    using select_target_range = std::vector<typename RangeGen<Field>::template target_range<range_factory,Communicator>>;
    template<typename Field>
    using select_source_range = std::vector<typename RangeGen<Field>::template source_range<range_factory,Communicator>>;

    template<typename Field>
    struct target_ranges
    {
        using ranges_type = select_target_range<Field>;
        using range_type = typename ranges_type::value_type;
        pattern_type* m_pattern;
        ranges_type m_ranges; 
    };

    template<typename Field>
    struct source_ranges
    {
        using ranges_type = select_source_range<Field>;
        using range_type = typename ranges_type::value_type;
        pattern_type* m_pattern;
        ranges_type m_ranges; 
    };

    using target_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<target_ranges, field_types>,std::tuple>;
    using source_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<source_ranges, field_types>,std::tuple>;

private: // members

    std::tuple<Fields*...> m_field_tuple;
    target_ranges_t m_target_ranges_tuple;
    source_ranges_t m_source_ranges_tuple;
    co_type m_co;
    std::vector<p_map_type> m_local_send_map;
    std::vector<p_map_type> m_local_recv_map;
    pattern_container_type& m_pattern;

    struct remote_exchanger
    {
        static co_handle fn(bulk_communication_object& ex)
        {
            return ex_impl(ex, std::make_index_sequence<sizeof...(Fields)>());
        }

        template<std::size_t... I>
        static co_handle ex_impl(bulk_communication_object& ex, std::index_sequence<I...>)
        {
            return ex.m_co.exchange(ex.m_pattern(*std::get<I>(ex.m_field_tuple))... );
        }
    };

public: // ctors

    bulk_communication_object(communicator_type comm, pattern_container_type& pattern, Fields&... fields)
    : m_field_tuple{&fields...}
    , m_co(comm)
    , m_local_send_map(pattern.size())
    , m_local_recv_map(pattern.size())
    , m_pattern(pattern)
    {
        // loop over domain-patterns and set up source ranges
        unsigned int d = 0;
        for (auto& p : pattern)
        {
            auto sit = p.send_halos().begin();
            while(sit != p.send_halos().end())
            {
                if (sit->first.mpi_rank == comm.rank())
                {
                    for (std::size_t i=0; i<sizeof...(Fields); ++i)
                    {
                        boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this,&p,d,&sit,&comm](auto i)
                        {
                            auto& f = *(std::get<decltype(i)::value>(m_field_tuple));
                            if (f.domain_id() == p.domain_id())
                            {
                                auto& source_r = std::get<decltype(i)::value>(m_source_ranges_tuple);
                                source_r.m_pattern = &p;
                                // loop over elements in index container
                                for (const auto& c : sit->second)
                                {
                                    source_r.m_ranges.emplace_back(comm, f, c.local().first(), c.local().last(), sit->first.tag); 
                                }
                            }
                        });
                    }
                    // remove the local send halos
                    m_local_send_map[d].insert(*sit);
                    sit = p.send_halos().erase(sit);
                }
                else
                    ++sit;
            }
            ++d;
        }

        // loop over domain-patterns and set up target ranges
        d = 0;
        for (auto& p : pattern)
        {
            auto rit = p.recv_halos().begin();
            while(rit != p.recv_halos().end())
            {
                if (rit->first.mpi_rank == comm.rank())
                {
                    for (std::size_t i=0; i<sizeof...(Fields); ++i)
                    {
                        boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this,&p,d,&rit,&comm](auto i)
                        {
                            auto& f = *(std::get<decltype(i)::value>(m_field_tuple));
                            if (f.domain_id() == p.domain_id())
                            {
                                auto& target_r = std::get<decltype(i)::value>(m_target_ranges_tuple);
                                target_r.m_pattern = &p;

                                // loop over elements in index container
                                for (const auto& c : rit->second)
                                {
                                    target_r.m_ranges.emplace_back(comm, f, c.local().first(), c.local().last(), rit->first.tag); 
                                    target_r.m_ranges.back().send();
                                }
                            }
                        });
                    }
                    // remove the local recv halos
                    m_local_recv_map[d].insert(*rit);
                    rit = p.recv_halos().erase(rit);
                }
                else
                    ++rit;
            }
            ++d;
        }
        
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get source ranges for this field
                auto& source_r = std::get<I::value>(m_source_ranges_tuple);
                for (auto& r : source_r.m_ranges)
                    r.recv();
            });
        }

    }

    void exchange()
    {
        // start communication for remote domains
        auto h = remote_exchanger::fn(*this);
        // loop over fields for putting
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get source ranges for this field
                auto& source_r = std::get<I::value>(m_source_ranges_tuple);
                using S = std::remove_reference_t<decltype(source_r)>;
                // get corresponding field
                auto& f = *(std::get<I::value>(m_field_tuple));
                using F = std::remove_reference_t<decltype(f)>;

                // get direct memory write access
                for (auto& r : source_r.m_ranges)
                    r.m_remote_range.start_source_epoch();

                // loop over the whole domain and put fields
                std::array<std::vector<typename S::range_type*>, F::dimension::value-1> filtered_ranges;

                typename F::coordinate_type coord;
                static constexpr auto Z = F::layout_map::template find<0>();
                for (auto z = -(long)f.offsets()[Z]; z<(long)f.extents()[Z]-f.offsets()[Z]; ++z)
                {
                    auto z_coord = coord;
                    z_coord[Z] = z;
                    filtered_ranges[0].clear();
                    for (auto& r : source_r.m_ranges)
                        if (z >= r.m_view.m_offset[Z] && z < r.m_view.m_offset[Z] + r.m_view.m_extent[Z])
                            filtered_ranges[0].push_back(&r);
                    static constexpr auto Y = F::layout_map::template find<1>();
                    for (auto y = -(long)f.offsets()[Y]; y<(long)f.extents()[Y]-f.offsets()[Y]; ++y)
                    {
                        auto y_coord = z_coord;
                        y_coord[Y] = y;
                        filtered_ranges[1].clear();
                        for (auto r : filtered_ranges[0])
                            if (y >= r->m_view.m_offset[Y] && y < r->m_view.m_offset[Y] + r->m_view.m_extent[Y])
                                filtered_ranges[1].push_back(r);
                        
                        static constexpr auto X = F::layout_map::template find<2>();
                        for (auto r : filtered_ranges[1])
                        {
                            auto x_coord = y_coord;
                            x_coord[X] = 0;//r->m_view.m_offset[X];
                            x_coord[Y] -= r->m_view.m_offset[Y];
                            x_coord[Z] -= r->m_view.m_offset[Z];
                            auto mem_loc = &(r->m_view(x_coord));
                            r->m_remote_range.put(r->m_pos,(const tl::ri::byte*)mem_loc);
                            ++r->m_pos;
                        }
                    }
                }
                
                // give up direct memory write access
                for (auto& r : source_r.m_ranges)
                {
                    r.m_pos = r.m_remote_range.begin();
                    r.m_remote_range.end_source_epoch();
                }
            });
        }
        // loop over fields for waiting
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get target ranges for this field
                auto& target_r = std::get<I::value>(m_target_ranges_tuple);
                // wait for communication to complete
                for (auto& r : target_r.m_ranges)
                    r.m_local_range.wait_at_target();
            });
        }
        h.wait();
    }
};

} // namespace structured

template<template <typename> class RangeGen, typename Communicator, typename Coordinate, typename DomainId, typename... Fields>
inline structured::bulk_communication_object<RangeGen, Communicator, Coordinate, DomainId, Fields...>
make_bulk_co(
        Communicator comm, 
        pattern_container<Communicator,structured::detail::grid<Coordinate>,DomainId>& pattern,
        Fields&... fields)
{
    return {comm, pattern, fields...};
}


} // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_EXCHANGE_HPP */
