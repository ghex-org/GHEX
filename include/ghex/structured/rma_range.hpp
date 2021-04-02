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
#include <iostream>
#include "../rma/chunk.hpp"
#include "./rma_range_iterator.hpp"

namespace gridtools {
namespace ghex {
namespace structured {
namespace detail {
// collection of helper template class to increment a range iterator
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

/** @brief An implementation of a multi-dimensional range which can be used on the device. This
  * type is used in the target and source ranges generated from the range generator. A list of
  * these types is passed to the range_factory.
  * @tparam Field the field type it is accessing */
template<typename Field>
struct rma_range
{
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using size_type = unsigned int;
    using fuse_components = std::integral_constant<bool,
        Field::has_components::value && (layout::at(dimension::value-1) == dimension::value-1)>;
    using iterator = range_iterator<rma_range>;

    Field m_field;
    coordinate m_offset;
    coordinate m_extent;
    coordinate m_begin;
    coordinate m_end;
    coordinate m_reduced_stride;
    size_type  m_size;
    size_type  m_num_elements;
    size_type  m_chunk_size;
    size_type  m_chunk_size_;

    template<typename Array>
    rma_range(const Field& f, const Array& offset, const Array& extent)
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
            m_size *= extent[i];
        }
        if (Field::has_components::value)
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = 0;
            m_extent[i] = f.num_components();
            m_begin[i] = 0;
            m_end[i] = m_extent[i]-1;
            m_size *= m_extent[i];
        }
        else
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            m_size *= extent[i];
        }

        m_end[I] = m_extent[I];
        m_num_elements = m_size;
        m_size  /= m_extent[I];
        m_chunk_size_ = m_extent[I];
        m_chunk_size = m_extent[I] * sizeof(value_type);

        m_reduced_stride[I] = 1;
        auto prod = m_reduced_stride[I];
        for (unsigned int d = dimension::value-1; d > 0; --d)
        {
            const auto i = layout::find(d-1);
            m_reduced_stride[i] = prod;
            prod *= m_extent[i];
        }
    }

    rma_range(const rma_range&) = default;
    rma_range(rma_range&&) = default;

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
    rma::chunk<value_type> get_chunk(const coordinate& coord) const noexcept {
        return {const_cast<value_type*>(ptr(coord)), m_chunk_size_};
    }

    GT_HOST_DEVICE
    void inc(size_type& index, int n, coordinate& coord) const noexcept {
        if (n < 0 && (size_type)(-n) > index)
        {
            coord = m_begin;
            index = 0;
            return;
        }
        index += n;
        if (index >= m_size)
        {
            coord = m_end;
            index = m_size;
        }
        else
        {
            auto idx = index;
            static constexpr auto I = layout::find(dimension::value-1);
            coord[I] = 0;
            for (unsigned int d = 0; d < dimension::value-1; ++d)
            {
                const auto i = layout::find(d);
                coord[i] = idx / m_reduced_stride[i];
                idx -= coord[i] * m_reduced_stride[i];
            }
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
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP */
