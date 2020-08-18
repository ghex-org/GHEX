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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_FIELD_VIEW_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_FIELD_VIEW_HPP

#include <cstring>
#include <vector>
#include <gridtools/common/host_device.hpp>

#include "./rma_range_iterator.hpp"
#include "../transport_layer/ri/access_guard.hpp"
#include "../common/utils.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

namespace detail {

#ifdef __CUDACC__

template<typename SourceRange, typename TargetRange>
__global__ void put_device_to_device_kernel(SourceRange sr, TargetRange tr, unsigned int size)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size)
    {
        using T = typename SourceRange::value_type;
        auto it = sr.begin();
        it += index;
        auto sr_chunk = *it;
        auto tr_chunk = *(tr.begin() + index);

        const unsigned int num_elements = sr_chunk.size()/sizeof(T);

        for (unsigned int i=0; i<num_elements; ++i)
        {
            *((T*)(tr_chunk.data())+i) = *((T*)(sr_chunk.data())+i);
        }
    }
}

#endif


template<unsigned int Dim, unsigned int D, typename Layout>
struct inc_coord
{
    template<typename Coord>
    GT_FUNCTION
    static void fn(Coord& coord, const Coord& ext) noexcept
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
    GT_FUNCTION
    static void fn(Coord& coord, const Coord& ext) noexcept
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
    GT_FUNCTION
    static void fn(Coord& coord, const Coord& ext) noexcept
    {
        static constexpr auto I = Layout::find(Dim-1);
        for (unsigned int i = 0; i < Dim; ++i) coord[i] = ext[i] - 1;
        coord[I] = ext[I];
    }
};
} // namespace detail


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
    using iterator = range_iterator<field_view>;

    Field m_field;
    rma_data_t m_rma_data;
    coordinate m_offset;
    coordinate m_extent;
    coordinate m_begin;
    coordinate m_end;
    coordinate m_reduced_stride;
    size_type  m_size;
    size_type  m_chunk_size;

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
        m_chunk_size = m_extent[I] * sizeof(value_type);
    }

    field_view(const field_view&) = default;
    field_view(field_view&&) = default;

    GT_FUNCTION
    iterator  begin() { return {this, 0, m_begin}; }
    GT_FUNCTION
    iterator  end()   { return {this, m_size, m_end}; }
    
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

    GT_FUNCTION
    tl::ri::chunk get_chunk(const coordinate& coord) const noexcept {
        return {const_cast<tl::ri::byte*>(reinterpret_cast<const tl::ri::byte*>(ptr(coord))), m_chunk_size};
    }

    GT_FUNCTION
    size_type inc(size_type index, size_type n, coordinate& coord) const noexcept {
        if (n < 0 && -n > index)
        {
            coord = m_begin;
            return 0;
        }
        index += n;
        if (index >= m_size)
        {
            coord = m_end;
            return m_size;
        }
        else
        {
            auto idx = index;
            static constexpr auto I = layout::find(dimension::value-1);
            coord[I] = 0;
            for (unsigned int d = 0; d < dimension::value; ++d)
            {
                const auto i = layout::find(d);
                coord[i] = index / m_reduced_stride[i];
                index -= coord[i] * m_reduced_stride[i];
            }
            return idx;
        }
    }

    GT_FUNCTION
    size_type inc(size_type index, coordinate& coord) const noexcept {
        static constexpr auto I = layout::find(dimension::value-1);
        if (index + 1 >= m_size)
        {
            coord = m_end;
            return m_size;
        }
        else
        {
            coord[I] = 0;
            detail::inc_coord<dimension::value, 1, layout>::fn(coord, m_extent);
            return index + 1;
        }
    }

    // put from cpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::cpu) const
    {
        put_host_to_host(r, fuse_components{});
    }

    // put from gpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::cpu) const
    {
        put_device_to_host(r, fuse_components{});
    }

    // put from cpu to gpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::gpu) const
    {
        put_host_to_device(r, fuse_components{});
    }

    // put from gpu to gpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::gpu) const
    {
        //put_device_to_device(r, fuse_components{});
#ifdef __CUDACC__
        static constexpr unsigned int block_dim = 128;
        const unsigned int num_blocks = (m_size+block_dim-1)/block_dim;
        detail::put_device_to_device_kernel<<<num_blocks,block_dim>>>(*this, r, m_size);
#endif
    }

private:

    template<typename RemoteRange>
    void put_host_to_host(RemoteRange& r, std::false_type) const
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
    template<typename RemoteRange>
    void put_host_to_host(RemoteRange& r, std::true_type) const
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

    template<typename RemoteRange>
    void put_device_to_host(RemoteRange& r, std::false_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 1>::apply(
            [this,&it](auto... c)
            {
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size(), cudaMemcpyDeviceToHost);
                ++it;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }
    template<typename RemoteRange>
    void put_device_to_host(RemoteRange& r, std::true_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 2>::apply(
            [this,&it](auto... c)
            {
                const auto nc = m_field.num_components();
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size()*nc, cudaMemcpyDeviceToHost);
                it+=nc;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }

    template<typename RemoteRange>
    void put_host_to_device(RemoteRange& r, std::false_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 1>::apply(
            [this,&it](auto... c)
            {
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size(), cudaMemcpyHostToDevice);
                ++it;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }
    template<typename RemoteRange>
    void put_host_to_device(RemoteRange& r, std::true_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 2>::apply(
            [this,&it](auto... c)
            {
                const auto nc = m_field.num_components();
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size()*nc, cudaMemcpyHostToDevice);
                it+=nc;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }

    template<typename RemoteRange>
    void put_device_to_device(RemoteRange& r, std::false_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 1>::apply(
            [this,&it](auto... c)
            {
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size(), cudaMemcpyDeviceToDevice);
                ++it;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }
    template<typename RemoteRange>
    void put_device_to_device(RemoteRange& r, std::true_type) const
    {
#ifdef __CUDACC__
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 2>::apply(
            [this,&it](auto... c)
            {
                const auto nc = m_field.num_components();
                auto chunk_ = *it;
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size()*nc, cudaMemcpyDeviceToDevice);
                it+=nc;
            },
            m_begin, m_end);
#else
        r.begin(); // prevent compiler warning
#endif
    }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_FIELD_VIEW_HPP */
