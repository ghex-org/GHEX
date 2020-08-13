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
#include <algorithm>
#include <tuple>
#include <boost/mp11.hpp>
#include "../common/moved_bit.hpp"
#include "../common/utils.hpp"
#include "../communication_object_2.hpp"
#include "../bulk_communication_object.hpp"
#include "../transport_layer/ri/range_factory.hpp"
#include "./pattern.hpp"


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
        static constexpr auto C = Layout::find(I-1);
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
    template<typename Range, typename Coord>
    static inline void apply2(Range& r, Coord coord) noexcept
    {
        static constexpr auto C = Layout::find(I-1);
        for (int c = 0; c < r.m_view.m_extent[C]; ++c)
        {
            coord[C] = c;
            range_loop<Layout,D,I+1>::apply2(r,coord);
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
        using range_type = std::remove_cv_t<std::remove_reference_t<decltype(**ranges.begin())>>;
        static constexpr auto C = Layout::find(0);
        
        static thread_local std::array<std::vector<range_type*>, D-1> filtered_ranges;
        for (auto c = -(long)f.offsets()[C]; c<(long)f.extents()[C]-f.offsets()[C]; ++c)
        {
            coordinate_type coord{};
            coord[C] = c;
            
            filtered_ranges[0].clear();
            for (auto& r : ranges)
                if (c >= r->m_view.m_offset[C] && c < r->m_view.m_offset[C] + r->m_view.m_extent[C])
                    filtered_ranges[0].push_back(r);

            range_loop<Layout,D,2>::apply(f,filtered_ranges,coord);
        }
    }
    template<typename Field, typename Ranges>
    static inline void apply2(Ranges& ranges) noexcept
    {
        using coordinate_type = typename Field::coordinate_type;
        static constexpr auto C = Layout::find(0);
        for (auto& r : ranges)
        {
            for (int c = 0; c < r.m_view.m_extent[C]; ++c)
            {
                coordinate_type coord{};
                coord[C] = c;
                range_loop<Layout,D,2>::apply2(r,coord);
            }
        }
    }
};

template<typename Layout, std::size_t D>
struct range_loop<Layout, D, D>
{
    template<typename Field, typename Filtered, typename Coord>
    static inline void apply(const Field&, Filtered& filtered_ranges, Coord coord) noexcept
    {
        static constexpr auto C = Layout::find(D-1);
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
    template<typename Range, typename Coord>
    static inline void apply2(Range& r, Coord coord) noexcept
    {
        static constexpr auto C = Layout::find(D-1);
        coord[C] = 0;
        auto mem_loc = r.m_view.ptr(coord);
        r.m_remote_range.put(r.m_pos,(const tl::ri::byte*)mem_loc);
        ++r.m_pos;
    }
};

} // namespace detail

template<template <typename> class RangeGen, typename Pattern, typename... Fields>
class bulk_communication_object;

template<template <typename> class RangeGen, typename Communicator, typename Coordinate, typename DomainId, typename... Fields>
class bulk_communication_object<
    RangeGen, 
    pattern_container<Communicator,structured::detail::grid<Coordinate>,DomainId>,
    Fields...>
{
public: // member types
    using communicator_type = Communicator;
    using pattern_type = pattern_container<Communicator,structured::detail::grid<Coordinate>,DomainId>;
    using grid_type = typename pattern_type::grid_type;
    using domain_id_type = typename pattern_type::domain_id_type;
    using co_type = communication_object<communicator_type,grid_type,domain_id_type>;
    using co_handle = typename co_type::handle_type;
    template<typename Field>
    using buffer_info_type = typename co_type::template buffer_info_type<typename Field::arch_type, Field>;

private: // member types
    template<typename A>
    struct select_arch_q
    {
        template<typename Field>
        using fn = typename Field::template rebind_arch<A>;
    };

    template<typename Field>
    using select_range = typename RangeGen<Field>::range_type;
    
    using field_types = boost::mp11::mp_unique<boost::mp11::mp_list<Fields...>>;
    using buffer_info_types = boost::mp11::mp_transform<buffer_info_type, field_types>;
    using cpu_fields = boost::mp11::mp_transform_q<select_arch_q<cpu>,field_types>;
#ifdef __CUDACC__
    using gpu_fields = boost::mp11::mp_transform_q<select_arch_q<gpu>,field_types>;
    using all_fields = boost::mp11::mp_unique<boost::mp11::mp_append<cpu_fields,gpu_fields>>;
#else
    using all_fields = boost::mp11::mp_unique<cpu_fields>;
#endif
    using all_ranges = boost::mp11::mp_transform<select_range,all_fields>;
    using range_factory = tl::ri::range_factory<all_ranges>;

    template<typename Field>
    using select_target_range = std::vector<typename RangeGen<Field>::template target_range<
        range_factory,communicator_type>>;
    template<typename Field>
    using select_source_range = std::vector<typename RangeGen<Field>::template source_range<
        range_factory,communicator_type>>;


    template<typename Field>
    struct target_ranges
    {
        using ranges_type = select_target_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges;
    };

    template<typename Field>
    struct source_ranges
    {
        using ranges_type = select_source_range<Field>;
        using range_type = typename ranges_type::value_type;
        std::vector<ranges_type> m_ranges; 
        std::vector<std::vector<range_type*>> m_range_ptrs;
    };
    
    using field_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<std::vector,field_types>,std::tuple>;
    
    using buffer_info_container_t = boost::mp11::mp_rename<boost::mp11::mp_transform<std::vector,buffer_info_types>,std::tuple>;

    using target_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<target_ranges, field_types>,std::tuple>;
    using source_ranges_t = boost::mp11::mp_rename<boost::mp11::mp_transform<source_ranges, field_types>,std::tuple>;

private: // members

    communicator_type       m_comm;
    co_type                 m_co;
    pattern_type            m_remote_pattern;
    pattern_type            m_local_pattern;
    field_container_t       m_field_container_tuple;
    buffer_info_container_t m_buffer_info_container_tuple;
    target_ranges_t         m_target_ranges_tuple;
    source_ranges_t         m_source_ranges_tuple;
    moved_bit               m_moved;
    bool                    m_initialized = false;

public: // ctors

    bulk_communication_object(communicator_type comm, const pattern_type& pattern)
    : m_comm(comm)
    , m_co(comm)
    , m_remote_pattern(pattern)
    , m_local_pattern(pattern)
    {
        // prepare local and remote patterns
        // =================================

        // loop over all subdomains in pattern
        for (int n = 0; n<pattern.size(); ++n)
        {
            auto& r_p = m_remote_pattern[n];
            auto& l_p = m_local_pattern[n];
            
            // loop over send halos
            auto r_it = r_p.send_halos().begin();
            auto l_it = l_p.send_halos().begin();
            while (r_it != r_p.send_halos().end())
            {
                const auto local = remote_range_traits<RangeGen>::is_local(comm, r_it->first.mpi_rank);
                if (local != tl::ri::locality::remote)
                {
                    // remove local fields from remote pattern
                    r_it = r_p.send_halos().erase(r_it);
                    ++l_it;
                }
                else
                {
                    // remove remote fields from local pattern
                    l_it = l_p.send_halos().erase(l_it);
                    ++r_it;
                }
            }

            // loop over recv halos
            r_it = r_p.recv_halos().begin();
            l_it = l_p.recv_halos().begin();
            while (r_it != r_p.recv_halos().end())
            {
                const auto local = remote_range_traits<RangeGen>::is_local(comm, r_it->first.mpi_rank);
                if (local != tl::ri::locality::remote)
                {
                    // remove local fields from remote pattern
                    r_it = r_p.recv_halos().erase(r_it);
                    ++l_it;
                }
                else
                {
                    // remove remote fields from local pattern
                    l_it = l_p.recv_halos().erase(l_it);
                    ++r_it;
                }
            }
        }
    }

    bulk_communication_object(const bulk_communication_object&) = delete;
    bulk_communication_object(bulk_communication_object&&) = default;

public:

    template<typename Field>
    void add_field(const Field& f)
    {
        if (m_moved)
            throw std::runtime_error("error: trying to add a field to a CO which is moved");
        if (m_initialized)
            throw std::runtime_error("error: this CO has been initialized already");

        using f_cont_t  = std::vector<Field>;
        using t_range_t = target_ranges<Field>;
        using s_range_t = source_ranges<Field>;

        auto& f_cont  = std::get<f_cont_t>(m_field_container_tuple);
        auto& t_range = std::get<t_range_t>(m_target_ranges_tuple);
        auto& s_range = std::get<s_range_t>(m_source_ranges_tuple);

        // store field
        f_cont.push_back(f);
        s_range.m_ranges.resize(s_range.m_ranges.size()+1);
        s_range.m_range_ptrs.resize(s_range.m_range_ptrs.size()+1);
        t_range.m_ranges.resize(t_range.m_ranges.size()+1);

        // loop over patterns 
        for (auto& p : m_local_pattern)
        {
            // check if field has the right domain
            if (f.domain_id() == p.domain_id())
            {
                // loop over halos and set up source ranges
                for (auto h_it = p.send_halos().begin(); h_it != p.send_halos().end(); ++h_it)
                {
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto& c = *it;
                        s_range.m_ranges.back().emplace_back(
                            m_comm, f, c.local().first(), c.local().last(),
                            h_it->first.mpi_rank, h_it->first.tag); 
                    }
                }
                // loop over halos and set up target
                for (auto h_it = p.recv_halos().begin(); h_it != p.recv_halos().end(); ++h_it)
                {
                    for (auto it = h_it->second.rbegin(); it != h_it->second.rend(); ++it)
                    {
                        const auto local = remote_range_traits<RangeGen>::is_local(m_comm, h_it->first.mpi_rank);
                        const auto& c = *it;
                        t_range.m_ranges.back().emplace_back(
                            m_comm, f, c.local().first(), c.local().last(),
                            h_it->first.mpi_rank, h_it->first.tag, local); 
                    }
                }
            }
        }
    }
    
    template<typename... F>
    void add_fields(const F&... fs)
    {
        auto fields = std::make_tuple(fs...);
        for (std::size_t i=0; i<sizeof...(F); ++i)
        {
            boost::mp11::mp_with_index<sizeof...(F)>(i,
            [this,&fields](auto i) {
                // get the field Index 
                using I = decltype(i);
                add_field(std::get<I::value>(fields));
            });
        }
    }

public:

    void init()
    {
        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get the field type
                using F = boost::mp11::mp_at<field_types, I>;
                // get field layout
                using Layout = typename F::layout_map;
                // get source and target ranges for fields of type F
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                auto& bi_cont = std::get<I::value>(m_buffer_info_container_tuple);
                auto& f_cont  = std::get<I::value>(m_field_container_tuple);
                // add remote exchange
                for (auto& f : f_cont)
                    bi_cont.push_back( m_remote_pattern(f) );
                // complete the handshake
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                    {
                        r.send();
                    }
                for (unsigned int k=0; k<s_range.m_ranges.size(); ++k)
                {
                    auto& s_vec = s_range.m_ranges[k];
                    auto& s_ptrs = s_range.m_range_ptrs[k];
                    for (auto& r : s_vec)
                    {
                        r.recv();
                        s_ptrs.push_back(&r);
                    }
                    
                    // sort s_ptrs
                    std::sort(s_ptrs.begin(), s_ptrs.end(), [](const auto& lhs, const auto& rhs)
                    {
                        return lhs->m_view.m_offset[Layout::find(F::dimension::value-1)] < 
                            rhs->m_view.m_offset[Layout::find(F::dimension::value-1)];
                    });
                }
            });
        }
        
        m_initialized = true;
    }

private:     
    co_handle exchange_remote()
    {
        return exchange_remote(std::make_index_sequence<boost::mp11::mp_size<field_types>::value>());
    }

    template<std::size_t... I>
    co_handle exchange_remote(std::index_sequence<I...>)
    {
        return m_co.exchange(std::make_pair(std::get<I>(m_buffer_info_container_tuple).begin(),
            std::get<I>(m_buffer_info_container_tuple).end())...);
    }

public:

    auto exchange()
    {
        if (!m_initialized) init();

        // start remote exchange
        auto h = exchange_remote();
        
        // loop over Fields
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get target ranges for fields and give remotes access
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                        r.m_local_range.end_target_epoch();
            });
        }
        // loop over fields for putting
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get the field type
                using F = boost::mp11::mp_at<field_types, I>;
                // get field layout
                using Layout = typename F::layout_map;
                // get source range
                auto& s_range = std::get<I::value>(m_source_ranges_tuple);
                //// A): loop over entire field
                //// get corresponding fields vector
                //auto& f_vec   = std::get<I::value>(m_field_container_tuple);
                // loop over field vector (all fields of type F)
                for (unsigned int k=0; k<s_range.m_ranges.size(); ++k)
                {
                    auto& s_vec  = s_range.m_ranges[k];
                    // get direct memory write access
                    for (auto& r : s_vec)
                        r.m_remote_range.start_source_epoch();
                
                    // put data
                    // ========

                    //// A): loop over entire field
                    //auto& s_ptrs = s_range.m_range_ptrs[k];
                    //auto& f      = f_vec[k];
                    //detail::range_loop<Layout, F::dimension::value, 1>::apply(f, s_ptrs);

                    // B): loop over each range seperately
                    detail::range_loop<Layout, F::dimension::value, 1>::template apply2<F>(s_vec);
                    
                    // give up direct memory write access
                    for (auto& r : s_vec)
                    {
                        r.m_pos = r.m_remote_range.begin();
                        r.m_remote_range.end_source_epoch();
                    }
                }
            });
        }
        // loop over fields for waiting
        for (std::size_t i=0; i<boost::mp11::mp_size<field_types>::value; ++i)
        {
            boost::mp11::mp_with_index<boost::mp11::mp_size<field_types>::value>(i,
            [this](auto i) {
                // get the field Index 
                using I = decltype(i);
                // get target ranges and wait
                auto& t_range = std::get<I::value>(m_target_ranges_tuple);
                for (auto& t_vec : t_range.m_ranges)
                    for (auto& r : t_vec)
                        r.m_local_range.start_target_epoch();
            });
        }

        // wait for remote exchange
        //h.wait();
        return h;
    }
};

} // namespace structured

} // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_EXCHANGE_HPP */

