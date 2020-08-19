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
#include <iostream>

#include "../common/utils.hpp"
#include "../cuda_utils/stream.hpp"
#include "../rma/chunk.hpp"
#include "../rma/access_guard.hpp"
#include "./rma_range_iterator.hpp"

namespace gridtools {
namespace ghex {
namespace structured {
namespace detail {
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
    using guard_type = rma::access_guard;
    using guard_view_type = rma::access_guard_view;
    using rma_data_t = typename Field::rma_data_t;
    using size_type = unsigned int;
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
    size_type  m_chunk_size_;

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

template<typename SourceField, typename TargetField>
using cpu_to_cpu = std::integral_constant<bool,
    std::is_same<typename SourceField::arch_type, gridtools::ghex::cpu>::value &&
    std::is_same<typename TargetField::arch_type, gridtools::ghex::cpu>::value>;
template<typename SourceField, typename TargetField>
using cpu_to_gpu = std::integral_constant<bool,
    std::is_same<typename SourceField::arch_type, gridtools::ghex::cpu>::value &&
    std::is_same<typename TargetField::arch_type, gridtools::ghex::gpu>::value>;
template<typename SourceField, typename TargetField>
using gpu_to_cpu = std::integral_constant<bool,
    std::is_same<typename SourceField::arch_type, gridtools::ghex::gpu>::value &&
    std::is_same<typename TargetField::arch_type, gridtools::ghex::cpu>::value>;
template<typename SourceField, typename TargetField>
using gpu_to_gpu = std::integral_constant<bool,
    std::is_same<typename SourceField::arch_type, gridtools::ghex::gpu>::value &&
    std::is_same<typename TargetField::arch_type, gridtools::ghex::gpu>::value>;

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    cpu_to_cpu<SourceField,TargetField>::value && !field_view<SourceField>::fuse_components::value>
put(field_view<SourceField>& s, field_view<TargetField>& t)
{
    using sv_t = field_view<SourceField>;
    using coordinate = typename sv_t::coordinate;
    gridtools::ghex::detail::for_loop<
        sv_t::dimension::value,
        sv_t::dimension::value,
        typename sv_t::layout, 1>::
    apply([&s,&t](auto... c)
    {
        std::memcpy(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size);
    },
    s.m_begin, s.m_end);
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    cpu_to_cpu<SourceField,TargetField>::value && field_view<SourceField>::fuse_components::value>
put(field_view<SourceField>& s, field_view<TargetField>& t)
{
    using sv_t = field_view<SourceField>;
    using coordinate = typename sv_t::coordinate;
    const auto nc = s.m_field.num_components();
    gridtools::ghex::detail::for_loop<
        sv_t::dimension::value,
        sv_t::dimension::value,
        typename sv_t::layout, 2>::
    apply([&s,&t,nc](auto... c)
    {
        std::memcpy(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size*nc);
    },
    s.m_begin, s.m_end);
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    cpu_to_gpu<SourceField,TargetField>::value>
put(field_view<SourceField>&, field_view<TargetField>&)
{
    // TODO
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    gpu_to_cpu<SourceField,TargetField>::value>
put(field_view<SourceField>&, field_view<TargetField>&)
{
    // TODO
}

#ifdef __CUDACC__
template<typename SourceRange, typename TargetRange>
__global__ void put_device_to_device_kernel(SourceRange sr, TargetRange tr)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < sr.m_size)
    {
        auto s_it = sr.begin();
        s_it += index;
        auto s_chunk = *s_it;
        auto t_it = tr.begin();
        t_it += index;
        auto t_chunk = *t_it;
        memcpy(t_chunk.data(), s_chunk.data(), s_chunk.bytes());
    }
}
#endif

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    gpu_to_gpu<SourceField,TargetField>::value>
put(field_view<SourceField>& s, field_view<TargetField>& t)
{
#ifdef __CUDACC__
    cuda::stream st;
    static constexpr unsigned int block_dim = 128;
    const unsigned int num_blocks = (s.m_size+block_dim-1)/block_dim;
    put_device_to_device_kernel<<<num_blocks,block_dim,0,st>>>(s, t);
    st.sync();
#endif
}

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_FIELD_VIEW_HPP */
