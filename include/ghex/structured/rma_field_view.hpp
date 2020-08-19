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

#include "./rma_range_iterator.hpp"
#include "../transport_layer/ri/access_guard.hpp"
#include "../common/utils.hpp"
#include "../cuda_utils/stream.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

namespace detail {
#ifdef __CUDACC__
#include <stdio.h>
__global__ void print_kernel() {
    printf("Hello from block %d, thread %d\n", blockIdx.x, threadIdx.x);
}
#endif
// does not yet work
#ifdef __CUDACC__
template<typename SourceRange, typename TargetRange>
__global__ void put_device_to_device_kernel(SourceRange sr, TargetRange& tr, unsigned int size)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    if (index < size)
    {
        printf("  in kernel %d %d\n", blockIdx.x, threadIdx.x);
        printf("    extent = %d %d %d\n", sr.m_extent[0], sr.m_extent[1], sr.m_extent[2]);

        auto it = sr.begin();
        //printf("    extent = %d %d %d\n", it.m_range->m_extent[0], it.m_range->m_extent[1], it.m_range->m_extent[2]);

        //printf("    it = %d %d %d (%d)\n", it.m_coord[0], it.m_coord[1], it.m_coord[2], it.m_index);
        it += index;
        const auto q = it.index();
        printf("    it = %d %d %d   %d\n", it.m_coord[0], it.m_coord[1], it.m_coord[2], it.m_index);
        printf("    it = %d %d %d   %d\n", it.m_coord[0], it.m_coord[1], it.m_coord[2], q);
        auto sr_chunk = *it;

        printf("    loc = %d \n", reinterpret_cast<unsigned long int>(sr_chunk.data()));
        
        auto it2 = tr.begin();
        //auto tr_chunk = *(tr.begin() + index);
        //printf("    loc = %d \n", reinterpret_cast<unsigned long int>(tr_chunk.data()));
        

        //using T = typename SourceRange::value_type;
        //auto it = sr.begin();
        //it += index;
        //auto sr_chunk = *it;
        //auto tr_chunk = *(tr.begin() + index);

        //const unsigned int num_bytes = sr_chunk.size(); // sizeof(T);

        //printf("copying %d bytes, from %d to %d \n", num_elements,
        //    reinterpret_cast<unsigned long int>(sr_chunk.data()),
        //    reinterpret_cast<unsigned long int>(tr_chunk.data())
        //        );

        ///*for (unsigned int i=0; i<num_elements; ++i)
        //{
        //    *((T*)(tr_chunk.data())+i) = *((T*)(sr_chunk.data())+i);
        //}*/
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
            //m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }
        if (Field::has_components::value)
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = 0;
            m_extent[i] = f.num_components();
            m_begin[i] = 0;
            m_end[i] = m_extent[i]-1;
            //m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= m_extent[i];
        }
        else
        {
            unsigned int i = dimension::value-1;
            m_offset[i] = offset[i];
            m_extent[i] = extent[i];
            m_begin[i] = 0;
            m_end[i] = extent[i]-1;
            //m_reduced_stride[i] = m_field.byte_strides()[i] / m_field.extents()[I];
            m_size *= extent[i];
        }

        m_end[I] = m_extent[I];
        m_size  /= m_extent[I];
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
    tl::ri::chunk get_chunk(const coordinate& coord) const noexcept {
        return {const_cast<tl::ri::byte*>(reinterpret_cast<const tl::ri::byte*>(ptr(coord))), m_chunk_size};
    }

    GT_HOST_DEVICE
    void inc(size_type& index, size_type n, coordinate& coord) const noexcept {
        if (n < 0 && -n > index)
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

    // put from cpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::cpu) const
    {
        //put_host_to_host(r, fuse_components{});
    }

    // put from gpu to cpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::cpu) const
    {
        //put_device_to_host(r, fuse_components{});
    }

    // put from cpu to gpu
    template<typename RemoteRange>
    void put(RemoteRange& r, gridtools::ghex::cpu, gridtools::ghex::gpu) const
    {
        //put_host_to_device(r, fuse_components{});
    }

//    // put from gpu to gpu
//    template<typename RemoteRange>
//    void put(RemoteRange& r, gridtools::ghex::gpu, gridtools::ghex::gpu) const
//    {
//        std::cout << "putting range" << std::endl;
//#ifdef __CUDACC__
////        // does not yet work
////        //cuda::stream s;
//        static constexpr unsigned int block_dim = 128;
//        const unsigned int num_blocks = (m_size+block_dim-1)/block_dim;
//        std::cout << "num lines = " << m_size << ", using " << num_blocks << " blocks" << std::endl;
////        //detail::put_device_to_device_kernel<<<num_blocks,block_dim,0,s>>>(*this, r, m_size);
//        detail::put_device_to_device_kernel<<<num_blocks,block_dim>>>(*this, r, m_size);
////        //s.sync();
//        cudaDeviceSynchronize();
//#endif
//        put_device_to_device(r, fuse_components{});
//#ifdef __CUDACC__
//        cudaDeviceSynchronize();
//#endif
//        std::cout << "putting range done" << std::endl;
//    }

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
        cuda::stream s;
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 1>::apply(
            [this,&it,&s](auto... c)
            {
                auto chunk_ = *it;
                //cudaMemcpyAsync(chunk_.data(), ptr(coordinate{c...}), chunk_.size(), cudaMemcpyDeviceToDevice, s);
                cudaMemcpy(chunk_.data(), ptr(coordinate{c...}), chunk_.size(), cudaMemcpyDeviceToDevice);
                ++it;
            },
            m_begin, m_end);
        //s.sync();
#else
        r.begin(); // prevent compiler warning
#endif
    }
    template<typename RemoteRange>
    void put_device_to_device(RemoteRange& r, std::true_type) const
    {
#ifdef __CUDACC__
        cuda::stream s;
        auto it = r.begin();
        gridtools::ghex::detail::for_loop<dimension::value, dimension::value, layout, 2>::apply(
            [this,&it,&s](auto... c)
            {
                const auto nc = m_field.num_components();
                auto chunk_ = *it;
                cudaMemcpyAsync(chunk_.data(), ptr(coordinate{c...}), chunk_.size()*nc, cudaMemcpyDeviceToDevice, s);
                it+=nc;
            },
            m_begin, m_end);
        s.sync();
#else
        r.begin(); // prevent compiler warning
#endif
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
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    gpu_to_cpu<SourceField,TargetField>::value>
put(field_view<SourceField>&, field_view<TargetField>&)
{
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
        memcpy(t_chunk.data(), s_chunk.data(), s_chunk.size());
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
