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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP

#include <cstring>
#include <vector>
#include <gridtools/common/host_device.hpp>

#include "../transport_layer/ri/types.hpp"
#include "../transport_layer/ri/access_guard.hpp"
#include "../common/utils.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

namespace range_detail {

template<typename Layout, std::size_t D, std::size_t I>
struct range_loop
{
    template<typename Range, typename Coord>
    static inline void apply(Range& r, Coord coord) noexcept
    {
        static constexpr auto C = Layout::find(I-1);
        for (int c = 0; c < r.m_view.m_extent[C]; ++c)
        {
            coord[C] = c;
            range_loop<Layout,D,I+1>::apply(r,coord);
        }
    }
};

template<typename Layout, std::size_t D>
struct range_loop<Layout, D, 1>
{
    template<typename Range>
    static inline void apply(Range& r) noexcept
    {
        using Field = typename Range::field_type;
        using coordinate_type = typename Field::coordinate_type;
        static constexpr auto C = Layout::find(0);
        for (int c = 0; c < r.m_view.m_extent[C]; ++c)
        {
            coordinate_type coord{};
            coord[C] = c;
            range_loop<Layout,D,2>::apply(r,coord);
        }
    }
};

template<typename Layout, std::size_t D>
struct range_loop<Layout, D, D>
{
    template<typename Range, typename Coord>
    static inline void apply(Range& r, Coord coord) noexcept
    {
        static constexpr auto C = Layout::find(D-1);
        coord[C] = 0;
        auto mem_loc = r.m_view.ptr(coord);
        r.m_remote_range.put(r.m_pos,(const tl::ri::byte*)mem_loc);
        ++r.m_pos;
    }
};

template<typename Range>
inline void put_loop(Range& r)
{
    range_loop<typename Range::field_type::layout_map, Range::field_type::dimension::value, 1>::apply(r);
}

} // namespace detail

template<typename Range>
struct range_iterator
{
    using coordinate = typename Range::coordinate;
    using chunk = tl::ri::chunk;
    using size_type = tl::ri::size_type;

    Range*      m_range;
    size_type   m_index;
    coordinate  m_coord;

    range_iterator(Range* r, size_type idx, const coordinate& coord)
    : m_range{r}
    , m_index{idx}
    , m_coord{coord}
    {}
    range_iterator(const range_iterator&) = default;
    range_iterator(range_iterator&&) = default;

    chunk     operator*() const noexcept { return m_range->get_chunk(m_coord); }
    void      operator++() noexcept { m_index = m_range->inc(m_index, m_coord); }
    void      operator--() noexcept { m_index = m_range->inc(m_index, -1, m_coord); }
    void      operator+=(size_type n) noexcept { m_index = m_range->inc(m_index, n, m_coord); }
    size_type sub(const range_iterator& other) const { return m_index - other.m_index; }
    bool      equal(const range_iterator& other) const { return m_index == other.m_index; }
    bool      lt(const range_iterator& other) const { return m_index < other.m_index; }
};

template<typename Field>
struct field_view
{
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = tl::ri::access_guard;
    using guard_view_type = tl::ri::access_guard_view;
    using rma_data_t = typename Field::rma_data_t;
    using size_type = tl::ri::size_type;
    using fuse_components = std::integral_constant<bool,
        Field::has_components::value && (layout::at(dimension::value-1) == dimension::value-1)>;

    Field m_field;
    rma_data_t m_rma_data;
    coordinate m_offset;
    coordinate m_extent;
    coordinate m_begin;
    coordinate m_end;
    coordinate m_reduced_stride;
    size_type  m_size;

    template<typename Array>
    field_view(const Field& f, const Array& offset, const Array& extent)
    : m_field(f)
    {
        static constexpr auto I = layout::find(dimension::value-1);
        m_size = 1;
        for (unsigned int i=0; i<dimension::value-1; ++i)
        {
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }
        if (Field::has_components::value)
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = 0;
            m_extent[i] = f.num_components();
            m_begin[i] = 0;
            m_end[i] = m_extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= m_extent[i];
        }
        else
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }

        m_end[I] = m_extent[I];
        m_size  /= m_extent[I];
    }

    field_view(const field_view&) = default;
    field_view(field_view&&) = default;
    
    GT_FUNCTION
    value_type& operator()(const coordinate& x) {
        return m_field(x+m_offset);
    }
    GT_FUNCTION
    const value_type& operator()(const coordinate& x) const {
        return m_field(x+m_offset);
    }
    
    GT_FUNCTION
    value_type* ptr(const coordinate& x) {
        return m_field.ptr(x+m_offset);
    }
    GT_FUNCTION
    const value_type* ptr(const coordinate& x) const {
        return m_field.ptr(x+m_offset);
    }
    // put from cpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::cpu) const
    {
        put(r, gridtools::ghex::cpu{}, gridtools::ghex::cpu{}, fuse_components{});
    }

    // put from gpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::cpu) const
    {
    }

    // put from cpu to gpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::gpu) const
    {
    }

    // put from gpu to gpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::gpu) const
    {
    }

private:

    // put from cpu to cpu: normal implementation
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::cpu, std::false_type) const
    {
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 1>::apply(
            [this,&it](auto... c)
            {
                auto chunk_ = *it;
                std::memcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size());
                ++it;
            },
            m_begin, m_end);
    }
    // put from cpu to cpu: fuse component loop
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::cpu, std::true_type) const
    {
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 2>::apply(
            [this,&it](auto... c)
            {
                const auto nc = m_field.num_components();
                auto chunk_ = *it;
                std::memcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size()*nc);
                it+=nc;
            },
            m_begin, m_end);
    }

};

namespace detail {
template<unsigned int Dim, unsigned int D, typename Layout>
struct inc_coord
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto I = Layout::find(Dim-D-1);
        if (coord[I] == ext[I] - 1)
        {
            coord[I] = 0;
            inc_coord<Dim, D + 1, Layout>::fn(coord, ext);
        }
        else
            coord[I] += 1;
    }
};
template<typename Layout>
struct inc_coord<3,1,Layout>
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto Y = Layout::find(1);
        static constexpr auto Z = Layout::find(0);
        const bool cond = coord[Y] < ext[Y] - 1;
        coord[Y]  = cond ? coord[Y] + 1 : 0;
        coord[Z] += cond ? 0 : 1;
    }
};
template<unsigned int Dim, typename Layout>
struct inc_coord<Dim, Dim, Layout>
{
    template<typename Coord>
    static inline void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto I = Layout::find(Dim-1);
        for (unsigned int i = 0; i < Dim; ++i) coord[i] = ext[i] - 1;
        coord[I] = ext[I];
    }
};
} // namespace detail

template<typename Field>
struct rma_range
{
    using arch_type = typename Field::arch_type;
    using view_type = field_view<Field>;
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = typename view_type::guard_type;
    using guard_view_type = typename view_type::guard_view_type;
    using size_type = tl::ri::size_type;
    using iterator = range_iterator<rma_range>;

    guard_view_type   m_guard;
    view_type         m_view;
    size_type         m_chunk_size;
    
    rma_range(const view_type& v, guard_type& g, tl::ri::locality loc) noexcept
    : m_guard{g, loc}
    , m_view{v}
    , m_chunk_size{(size_type)(m_view.m_extent[layout::find(dimension::value-1)] * sizeof(value_type))}
    {}
    
    rma_range(const rma_range&) = default;
    rma_range(rma_range&&) = default;

    iterator  begin() const { return {const_cast<rma_range*>(this), 0, m_view.m_begin}; }
    iterator  end()   const { return {const_cast<rma_range*>(this), m_view.m_size, m_view.m_end}; }
    size_type buffer_size() const { return m_chunk_size; }

    // these functions are called at the remote site upon deserializing and reconstructing the range
    // and can be used to allocate state
    void init(tl::ri::remote_host_)   
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data, m_guard.get_locality());
	    m_guard.init_remote();
    }
    void init(tl::ri::remote_device_)
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data, m_guard.get_locality());
	    m_guard.init_remote();
    }
    void exit(tl::ri::remote_host_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    void exit(tl::ri::remote_device_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    
    void start_local_epoch() { m_guard.start_local_epoch(); }
    void end_local_epoch()   { m_guard.end_local_epoch(); }

    void start_remote_epoch(tl::ri::remote_host_)   { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_host_)     { m_guard.end_remote_epoch(); }
    void start_remote_epoch(tl::ri::remote_device_) { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_device_)   { m_guard.end_remote_epoch(); }
    
    tl::ri::chunk get_chunk(const coordinate& coord) const noexcept {
        auto ptr = const_cast<tl::ri::byte*>(reinterpret_cast<const tl::ri::byte*>(&(m_view(coord))));
        return {ptr, m_chunk_size};
    }

    size_type inc(size_type index, size_type n, coordinate& coord) const noexcept {
        if (n < 0 && -n > index)
        {
            coord = m_view.m_begin;
            return 0;
        }
        index += n;
        if (index >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            auto idx = index;
            static constexpr auto I = layout::find(dimension::value-1);
            coord[I] = 0;
            for (unsigned int d = 0; d < dimension::value; ++d)
            {
                const auto i = layout::find(d);
                coord[i] = index / m_view.m_reduced_stride[i];
                index -= coord[i] * m_view.m_reduced_stride[i];
            }
            return idx;
        }
    }

    size_type inc(size_type index, coordinate& coord) const noexcept {
        static constexpr auto I = layout::find(dimension::value-1);
        if (index + 1 >= m_view.m_size)
        {
            coord = m_view.m_end;
            return m_view.m_size;
        }
        else
        {
            coord[I] = 0;
            detail::inc_coord<dimension::value, 1, layout>::fn(coord, m_view.m_extent);
            return index + 1;
        }
    }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP */
