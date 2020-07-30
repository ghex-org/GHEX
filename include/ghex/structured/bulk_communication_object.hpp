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
#ifndef INCLUDED_GHEX_STRUCTURED_BULK_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_STRUCTURED_BULK_COMMUNICATION_OBJECT_HPP

#include <vector>
#include <tuple>
#include <boost/mp11.hpp>
#include "../common/moved_bit.hpp"
#include "../common/utils.hpp"
#include "../communication_object_2.hpp"
#include "../bulk_communication_object.hpp"
#include "../transport_layer/ri/range_factory.hpp"
#include "./pattern.hpp"

//#include <iostream>

namespace gridtools {

namespace ghex {

namespace structured {

namespace detail {

template<typename Layout, std::size_t D, std::size_t I>
struct range_loop
{
    template<typename Field, typename Filtered, typename Coord>
    static inline void apply(const Field& f, Filtered& filtered_ranges, Coord coord) noexcept
    {
        static constexpr auto C = Layout::template find<I-1>();
        for (auto c = -(long)f.offsets()[C]; c<(long)f.extents()[C]-f.offsets()[C]; ++c)
        {
            coord[C] = c;
            
            filtered_ranges[I-1].clear();
            for (auto r : filtered_ranges[I-2])
                if (c >= r->m_view.m_offset[C] && c < r->m_view.m_offset[C] + r->m_view.m_extent[C])
                    filtered_ranges[I-1].push_back(r);
            
            range_loop<Layout,D,I+1>::apply(f,filtered_ranges,coord);
        }
    }
};

template<typename Layout, std::size_t D>
struct range_loop<Layout, D, 1>
{
    template<typename Field, typename Ranges>
    static inline void apply(const Field& f, Ranges& ranges) noexcept
    {
        using coordinate_type = typename Field::coordinate_type;
        using range_type = std::remove_cv_t<std::remove_reference_t<decltype(*ranges.begin())>>;
        static constexpr auto C = Layout::template find<0>();
        
        static thread_local std::array<std::vector<range_type*>, D-1> filtered_ranges;
        for (auto c = -(long)f.offsets()[C]; c<(long)f.extents()[C]-f.offsets()[C]; ++c)
        {
            coordinate_type coord{};
            coord[C] = c;
            
            filtered_ranges[0].clear();
            for (auto& r : ranges)
                if (c >= r.m_view.m_offset[C] && c < r.m_view.m_offset[C] + r.m_view.m_extent[C])
                    filtered_ranges[0].push_back(&r);

            range_loop<Layout,D,2>::apply(f,filtered_ranges,coord);
        }
    }
};

template<typename Layout, std::size_t D>
struct range_loop<Layout, D, D>
{
    template<typename Field, typename Filtered, typename Coord>
    static inline void apply(const Field&, Filtered& filtered_ranges, Coord coord) noexcept
    {
        static constexpr auto C = Layout::template find<D-1>();
        for (auto r : filtered_ranges[D-2])
        {
            auto coord_new = coord;
            for (unsigned int d=0; d<D; ++d)
                coord_new[d] -= (d==C) ? coord[d] : r->m_view.m_offset[d];
            auto mem_loc = r->m_view.ptr(coord_new);
            r->m_remote_range.put(r->m_pos,(const tl::ri::byte*)mem_loc);
            ++r->m_pos;
        }
    }
};

}

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
        ranges_type m_ranges; 
    };

    template<typename Field>
    struct source_ranges
    {
        using ranges_type = select_source_range<Field>;
        using range_type = typename ranges_type::value_type;
        ranges_type m_ranges; 
    };

    using target_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<target_ranges, field_types>,std::tuple>;
    using source_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<source_ranges, field_types>,std::tuple>;

private: // members

    std::tuple<Fields*...> m_field_tuple;
    target_ranges_t m_target_ranges_tuple;
    source_ranges_t m_source_ranges_tuple;
    co_type m_co;
    pattern_container_type m_pattern;
    moved_bit m_moved;

public: // ctors

    bulk_communication_object(communicator_type comm, pattern_container_type& pattern, Fields&... fields)
    : m_field_tuple{&fields...}
    , m_co(comm)
    , m_pattern(pattern)
    {
        // loop over domain-patterns and set up source ranges
        unsigned int d = 0;

        const auto max_tag = m_pattern.max_tag();
        for (auto& p : m_pattern)
        {
            auto sit = p.send_halos().begin();
            while(sit != p.send_halos().end())
            {
                if (remote_range_traits<RangeGen>::is_local(comm, sit->first.mpi_rank))
                {
                    for (std::size_t i=0; i<sizeof...(Fields); ++i)
                    {
                        boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this,&p,d,&sit,&comm,max_tag](auto i) mutable
                        {
                            auto& f = *(std::get<decltype(i)::value>(m_field_tuple));
                            if (f.domain_id() == p.domain_id())
                            {
                                auto& source_r = std::get<decltype(i)::value>(m_source_ranges_tuple);
                                // loop over elements in index container
                                //for (const auto& c : sit->second)
                                //{
                                for (auto it = sit->second.rbegin(); it != sit->second.rend(); ++it)
                                {
                                    const auto& c = *it;
                                    source_r.m_ranges.emplace_back(comm, f, c.local().first(), c.local().last(), sit->first.mpi_rank, sit->first.tag); 
                                }
                            }
                        });
                    }
                    // remove the local send halos
                    sit = p.send_halos().erase(sit);
                }
                else
                    ++sit;
            }
            ++d;
        }

        // loop over domain-patterns and set up target ranges
        d = 0;
        for (auto& p : m_pattern)
        {
            auto rit = p.recv_halos().begin();
            while(rit != p.recv_halos().end())
            {
                if (remote_range_traits<RangeGen>::is_local(comm, rit->first.mpi_rank))
                {
                    for (std::size_t i=0; i<sizeof...(Fields); ++i)
                    {
                        boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this,&p,d,&rit,&comm,&max_tag](auto i) mutable
                        {
                            auto& f = *(std::get<decltype(i)::value>(m_field_tuple));
                            if (f.domain_id() == p.domain_id())
                            {
                                auto& target_r = std::get<decltype(i)::value>(m_target_ranges_tuple);
                                // loop over elements in index container
                                //for (const auto& c : rit->second)
                                //{
                                for (auto it = rit->second.rbegin(); it != rit->second.rend(); ++it)
                                {
                                    const auto& c = *it;
                                    target_r.m_ranges.emplace_back(comm, f, c.local().first(), c.local().last(), rit->first.mpi_rank, rit->first.tag); 
                                    // start handshake
                                    target_r.m_ranges.back().send();
                                }
                            }
                        });
                    }
                    // remove the local recv halos
                    rit = p.recv_halos().erase(rit);
                }
                else
                    ++rit;
            }
            ++d;
        }
        
        // loop over source ranges
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get source ranges for this field
                auto& source_r = std::get<I::value>(m_source_ranges_tuple);
                // complete the handshake
                for (auto& r : source_r.m_ranges) r.recv();
            });
        }
    }

    bulk_communication_object(bulk_communication_object&& ) = default;

    ~bulk_communication_object()
    {
        if (!m_moved)
        {
            for (std::size_t i=0; i<sizeof...(Fields); ++i)
            {
                boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
                {
                    using I = decltype(i);
                    // get target ranges for this field
                    auto& target_r = std::get<I::value>(m_target_ranges_tuple);
                    // wait for communication to complete
                    for (auto& r : target_r.m_ranges)
                        r.release();
                });
            }
        }
    }
public: // member functions
    void exchange()
    {
        // start communication for remote domains
        auto h = exchange_remote();
        // loop over fields for granting remote access
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get target ranges for this field
                auto& target_r = std::get<I::value>(m_target_ranges_tuple);
                // wait for communication to complete
                for (auto& r : target_r.m_ranges)
                    r.m_local_range.end_target_epoch();
            });
        }
        // loop over fields for putting
        for (std::size_t i=0; i<sizeof...(Fields); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(Fields)>(i, [this](auto i)
            {
                using I = decltype(i);
                // get source ranges for this field
                auto& source_r = std::get<I::value>(m_source_ranges_tuple);
                // get corresponding field
                auto& f = *(std::get<I::value>(m_field_tuple));
                using F = std::remove_reference_t<decltype(f)>;

                // get direct memory write access
                for (auto& r : source_r.m_ranges)
                    r.m_remote_range.start_source_epoch();

                //// general loop
                //detail::range_loop<typename F::layout_map, F::dimension::value, 1>::apply(f, source_r.m_ranges);

                //// hard coded 3D loop
                //using S = std::remove_reference_t<decltype(source_r)>;
                //std::array<std::vector<typename S::range_type*>, F::dimension::value-1> filtered_ranges;
                //typename F::coordinate_type coord;
                //static constexpr auto Z = F::layout_map::template find<0>();
                //for (auto z = -(long)f.offsets()[Z]; z<(long)f.extents()[Z]-f.offsets()[Z]; ++z)
                //{
                //    auto z_coord = coord;
                //    z_coord[Z] = z;
                //    filtered_ranges[0].clear();
                //    for (auto& r : source_r.m_ranges)
                //        if (z >= r.m_view.m_offset[Z] && z < r.m_view.m_offset[Z] + r.m_view.m_extent[Z])
                //            filtered_ranges[0].push_back(&r);
                //    static constexpr auto Y = F::layout_map::template find<1>();
                //    for (auto y = -(long)f.offsets()[Y]; y<(long)f.extents()[Y]-f.offsets()[Y]; ++y)
                //    {
                //        auto y_coord = z_coord;
                //        y_coord[Y] = y;
                //        filtered_ranges[1].clear();
                //        for (auto r : filtered_ranges[0])
                //            if (y >= r->m_view.m_offset[Y] && y < r->m_view.m_offset[Y] + r->m_view.m_extent[Y])
                //                filtered_ranges[1].push_back(r);
                //        
                //        static constexpr auto X = F::layout_map::template find<2>();
                //        for (auto r : filtered_ranges[1])
                //        {
                //            auto x_coord = y_coord;
                //            x_coord[X] = 0;//r->m_view.m_offset[X];
                //            x_coord[Y] -= r->m_view.m_offset[Y];
                //            x_coord[Z] -= r->m_view.m_offset[Z];
                //            auto mem_loc = r->m_view.ptr(x_coord);
                //            r->m_remote_range.put(r->m_pos,(const tl::ri::byte*)mem_loc);
                //            ++r->m_pos;
                //        }
                //    }
                //}

                // loop over ranges - hard coded 3D
                for (auto& r : source_r.m_ranges)
                {
                    for (int z = 0; z < r.m_view.m_extent[2]; ++z)
                        for (int y = 0; y < r.m_view.m_extent[1]; ++y)
                        {
                            auto mem_loc = r.m_view.ptr(typename F::coordinate_type{0, y, z});
                            r.m_remote_range.put(r.m_pos,(const tl::ri::byte*)mem_loc);
                            ++r.m_pos;
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
                    //r.m_local_range.wait_at_target();
                    r.m_local_range.start_target_epoch();
            });
        }
        h.wait();
    }

private: // implementation details
    co_handle exchange_remote() { return exchange_remote(std::make_index_sequence<sizeof...(Fields)>()); }

    template<std::size_t... I>
    co_handle exchange_remote(std::index_sequence<I...>) { return m_co.exchange(m_pattern(*std::get<I>(m_field_tuple))... ); }
};

} // namespace structured

template<template <typename> class RangeGen, typename Communicator, typename Coordinate, typename DomainId, typename... Fields>
inline bulk_communication_object
make_bulk_co(Communicator comm, pattern_container<Communicator,structured::detail::grid<Coordinate>,DomainId>& pattern, Fields&... fields)
{
    using type = structured::bulk_communication_object<RangeGen, Communicator, Coordinate, DomainId, Fields...>;
    return {type{comm, pattern, fields...}};
}

} // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_EXCHANGE_HPP */
