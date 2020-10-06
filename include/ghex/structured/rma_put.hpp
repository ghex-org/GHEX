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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_PUT_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_PUT_HPP

#include "../common/utils.hpp"
#include "../cuda_utils/stream.hpp"
#include "./rma_range.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

// Here are different specializations of put functions. A put function takes 2 arguments (source and
// target range) and performs the RMA put.

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
    cpu_to_cpu<SourceField,TargetField>::value && !rma_range<SourceField>::fuse_components::value>
put(rma_range<SourceField>& s, rma_range<TargetField>& t
#ifdef __CUDACC__
    , cudaStream_t
#endif
)
{
    using sv_t = rma_range<SourceField>;
    using coordinate = typename sv_t::coordinate;

    auto dst = t.m_field.data();
    auto src = s.m_field.data();
    const auto dimx = s.m_field.extents()[0];
    const auto dimy = s.m_field.extents()[1];
    const int src_ox = s.m_offset[0]+s.m_field.offsets()[0];
    const int src_oy = s.m_offset[1]+s.m_field.offsets()[1];
    const int src_oz = s.m_offset[2]+s.m_field.offsets()[2];
    const int dst_ox = t.m_offset[0]+t.m_field.offsets()[0];
    const int dst_oy = t.m_offset[1]+t.m_field.offsets()[1];
    const int dst_oz = t.m_offset[2]+t.m_field.offsets()[2];
    for (int j=0; j<s.m_extent[1]; ++j)
        for (int k=0; k<s.m_extent[2]; ++k)
            for (int i=0; i<s.m_extent[0]; ++i)
            {
                dst[(k+dst_oz)*dimy*dimx + (j+dst_oy)*dimx + (i+dst_ox)] = 
                    src[(k+src_oz)*dimy*dimx + (j+src_oy)*dimx + (i+src_ox)];
            }

    // using sv_t = rma_range<SourceField>;
    // using coordinate = typename sv_t::coordinate;
    // using T = typename SourceField::value_type;
    // unsigned char* dst = (unsigned char*)t.m_field.data();
    // const unsigned char* src = (const unsigned char*)s.m_field.data();
    // const auto src_o = s.m_offset+s.m_field.offsets();
    // const auto dst_o = t.m_offset+t.m_field.offsets();
    // const auto src_strides = s.m_field.byte_strides();
    // const auto dst_strides = t.m_field.byte_strides();
    // for (int k=0; k<s.m_extent[2]; ++k)
    //     for (int j=0; j<s.m_extent[1]; ++j)
    //         for (int i=0; i<s.m_extent[0]; ++i)
    //         {
    //             *((T*)      (dst
    //                     +(i+dst_o[0])*dst_strides[0]
    //                     +(j+dst_o[1])*dst_strides[1]
    //                     +(k+dst_o[2])*dst_strides[2])) =
    //                 *((const T*)(src
    //                         +(i+src_o[0])*src_strides[0]
    //                         +(j+src_o[1])*src_strides[1]
    //                         +(k+src_o[2])*src_strides[2]));
    //         }

    // for (int k=0; k<s.m_extent[2]; ++k)
    //     for (int j=0; j<s.m_extent[1]; ++j)
    //     {
    //         auto tp = (dst
    //             +(  dst_o[0])*dst_strides[0]
    //             +(j+dst_o[1])*dst_strides[1]
    //             +(k+dst_o[2])*dst_strides[2]) ;
    //             auto sp = (src
    //                 +(  src_o[0])*src_strides[0]
    //                 +(j+src_o[1])*src_strides[1]
    //                 +(k+src_o[2])*src_strides[2]);
    //         std::memcpy(tp,sp,s.m_chunk_size);
    //     }

    // for (int k=0; k<s.m_extent[2]; ++k)
    //     for (int j=0; j<s.m_extent[1]; ++j)
    //     {
    //         auto tp = (T*)(dst
    //             +(  dst_o[0])*dst_strides[0]
    //             +(j+dst_o[1])*dst_strides[1]
    //             +(k+dst_o[2])*dst_strides[2]) ;
    //             auto sp = (const T*)(src
    //                 +(  src_o[0])*src_strides[0]
    //                 +(j+src_o[1])*src_strides[1]
    //                 +(k+src_o[2])*src_strides[2]);
    //         for (int i=0; i<s.m_extent[0]; ++i)
    //         {
    //             tp[i] = sp[i];
    //         }
    //     }

    // using sv_t = rma_range<SourceField>;
    // using coordinate = typename sv_t::coordinate;
    // using T = typename SourceField::value_type;
    // unsigned char* dst = (unsigned char*)t.m_field.data();
    // const unsigned char* src = (const unsigned char*)s.m_field.data();
    // const auto src_o = s.m_offset+s.m_field.offsets();
    // const auto dst_o = t.m_offset+t.m_field.offsets();
    // const auto src_strides = s.m_field.byte_strides();
    // const auto dst_strides = t.m_field.byte_strides();
    // for (int k=0; k<s.m_extent[2]; ++k)
    //     for (int j=0; j<s.m_extent[1]; ++j)
    //         for (int i=0; i<s.m_extent[0]; ++i)
    //         {
    //             *((T*)      (dst+dot(coordinate{i,j,k}+dst_o, dst_strides))) = 
    //                 *((const T*)(src+dot(coordinate{i,j,k}+src_o, src_strides)));
    //         }

    // gridtools::ghex::detail::for_loop<
    //     sv_t::dimension::value,
    //     sv_t::dimension::value,
    //     typename sv_t::layout, 1>::
    // apply([&s,&t](auto... c)
    // {
    //     std::memcpy(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size);
    //     // auto dst = t.ptr(coordinate{c...});
    //     // auto src = s.ptr(coordinate{c...});
    //     // for (unsigned int i=0; i<s.m_chunk_size_; ++i)
    //     // {
    //     //     dst[i] = src[i]; 
    //     // }
    // },
    // s.m_begin, s.m_end);

}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    cpu_to_cpu<SourceField,TargetField>::value && rma_range<SourceField>::fuse_components::value>
put(rma_range<SourceField>& s, rma_range<TargetField>& t
#ifdef __CUDACC__
    , cudaStream_t
#endif
)
{
    using sv_t = rma_range<SourceField>;
    using coordinate = typename sv_t::coordinate;
    const auto nc = s.m_field.num_components();
    gridtools::ghex::detail::for_loop<
        sv_t::dimension::value,
        sv_t::dimension::value,
        typename sv_t::layout, 2>::
    apply([&s,&t,nc](auto... c)
    {
        std::memcpy(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size*nc);
        // auto dst = t.ptr(coordinate{c...});
        // auto src = s.ptr(coordinate{c...});
        // for (unsigned int i=0; i<s.m_chunk_size_*nc; ++i)
        // {
        //     dst[i] = src[i]; 
        // }
    },
    s.m_begin, s.m_end);
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    cpu_to_gpu<SourceField,TargetField>::value>
put(rma_range<SourceField>& s, rma_range<TargetField>& t
#ifdef __CUDACC__
    , cudaStream_t st
#endif
)
{
#ifdef __CUDACC__
    using sv_t = rma_range<SourceField>;
    using coordinate = typename sv_t::coordinate;
    gridtools::ghex::detail::for_loop<
        sv_t::dimension::value,
        sv_t::dimension::value,
        typename sv_t::layout, 1>::
    apply([&s,&t,&st](auto... c)
    {
        cudaMemcpyAsync(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size, 
            cudaMemcpyHostToDevice, st);
    },
    s.m_begin, s.m_end);
#endif
}

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    gpu_to_cpu<SourceField,TargetField>::value>
put(rma_range<SourceField>& s, rma_range<TargetField>& t
#ifdef __CUDACC__
    , cudaStream_t st
#endif
)
{
#ifdef __CUDACC__
    using sv_t = rma_range<SourceField>;
    using coordinate = typename sv_t::coordinate;
    gridtools::ghex::detail::for_loop<
        sv_t::dimension::value,
        sv_t::dimension::value,
        typename sv_t::layout, 1>::
    apply([&s,&t,&st](auto... c)
    {
        cudaMemcpyAsync(t.ptr(coordinate{c...}), s.ptr(coordinate{c...}), s.m_chunk_size, 
            cudaMemcpyDeviceToHost, st);
    },
    s.m_begin, s.m_end);
#endif
}

#ifdef __CUDACC__
template<typename SourceRange, typename TargetRange>
__global__ void put_device_to_device_kernel(SourceRange sr, TargetRange tr)
{
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    //if (index < sr.m_size)
    if (index < sr.m_num_elements)
    {
        const auto line_index = index/sr.m_chunk_size_;
        const auto x = index - line_index*sr.m_chunk_size_;
        auto s_it = sr.begin();
        s_it += line_index;
        auto s_chunk = *s_it;
        auto t_it = tr.begin();
        t_it += line_index;
        auto t_chunk = *t_it;
        memcpy(&t_chunk[x], &s_chunk[x], sizeof(typename SourceRange::value_type));
        //t_chunk[x] = (const typename SourceRange::value_type &)s_chunk[x];
        //memcpy(t_chunk.data(), s_chunk.data(), s_chunk.bytes());
    }
}
#endif

template<typename SourceField, typename TargetField>
inline std::enable_if_t<
    gpu_to_gpu<SourceField,TargetField>::value>
put(rma_range<SourceField>& s, rma_range<TargetField>& t
#ifdef __CUDACC__
    , cudaStream_t st
#endif
)
{
#ifdef __CUDACC__
    static constexpr unsigned int block_dim = 128;
    const unsigned int num_blocks = (s.m_num_elements+block_dim-1)/block_dim;
    put_device_to_device_kernel<<<num_blocks,block_dim,0,st>>>(s, t);
#endif
}

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_PUT_HPP */
