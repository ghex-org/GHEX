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
#ifndef INCLUDED_GHEX_STRUCTURED_PACK_KERNELS_HPP
#define INCLUDED_GHEX_STRUCTURED_PACK_KERNELS_HPP

#include "./field_utils.hpp"
#include "../common/utils.hpp"
#include "../arch_traits.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

/** @brief Helper class to dispatch to CPU/GPU implementations of pack/unpack kernels
  * @tparam Arch Architecture type
  * @tparam LayoutMap Data layout map*/
template<typename Arch, typename LayoutMap>
struct serialization {
    template<typename PackIterationSpace>
    static void pack(PackIterationSpace&& pack_is, void*) {
        using coordinate_type = typename PackIterationSpace::coordinate_t;
        ::gridtools::ghex::detail::for_loop<4,4,LayoutMap>::template apply(
            [&pack_is](int x, int y, int z, int c) {
                pack_is.buffer(coordinate_type{x,y,z,c}) = pack_is.data(coordinate_type{x,y,z,c});},
            pack_is.m_data_is.m_first,
            pack_is.m_data_is.m_last);
    }

    template<typename UnPackIterationSpace>
    static void unpack(UnPackIterationSpace&& unpack_is, void*) {
        using coordinate_type = typename UnPackIterationSpace::coordinate_t;
        ::gridtools::ghex::detail::for_loop<4,4,LayoutMap>::template apply(
            [&unpack_is](int x, int y, int z, int c) {
                unpack_is.data(coordinate_type{x,y,z,c}) = unpack_is.buffer(coordinate_type{x,y,z,c});
            },
            unpack_is.m_data_is.m_first,
            unpack_is.m_data_is.m_last);
    }
};

#ifdef __CUDACC__
template<typename Layout, typename PackIterationSpace>
__global__ void pack_kernel(PackIterationSpace pack_is, unsigned int num_elements) {
    using value_type = typename PackIterationSpace::value_t;
    using coordinate_type = typename PackIterationSpace::coordinate_t;
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i=0; i<num_elements; ++i) {
        if (index*num_elements+i < pack_is.m_buffer_desc.m_size) {
            // compute local coordinate
            coordinate_type local_coordinate;
            ::gridtools::ghex::structured::detail::compute_coordinate<4>::template apply<Layout>(
                pack_is.m_data_is.m_local_strides, local_coordinate, index*num_elements+i);
            // add offset
            const coordinate_type x = local_coordinate + pack_is.m_data_is.m_first;
            // assign
            pack_is.buffer(x) = pack_is.data(x);
        }
    }
}

template<typename Layout, typename UnPackIterationSpace>
__global__ void unpack_kernel(UnPackIterationSpace unpack_is, unsigned int num_elements) {
    using value_type = typename UnPackIterationSpace::value_t;
    using coordinate_type = typename UnPackIterationSpace::coordinate_t;
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i=0; i<num_elements; ++i) {
        if (index*num_elements+i < unpack_is.m_buffer_desc.m_size) {
            // compute local coordinate
            coordinate_type local_coordinate;
            ::gridtools::ghex::structured::detail::compute_coordinate<4>::template apply<Layout>(
                unpack_is.m_data_is.m_local_strides, local_coordinate, index*num_elements+i);
            // add offset
            const coordinate_type x = local_coordinate + unpack_is.m_data_is.m_first;
            // assign
            unpack_is.data(x) = unpack_is.buffer(x);
        }
    }
}

template<typename LMap>
struct serialization<::gridtools::ghex::gpu,LMap> {
    // kernels use a 1-D compute grid with block-dim threads per block
    static constexpr std::size_t block_dim = 128;
    // number of values to pack/unpack per thread
    static constexpr std::size_t elements_per_thread = 1;

    template<typename PackIterationSpace>
    static void pack(PackIterationSpace&& pack_is, void* arg) {
        using coordinate_type = typename PackIterationSpace::coordinate_t;
        auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
        const std::size_t num_threads = (pack_is.m_buffer_desc.m_size + elements_per_thread-1)
            /elements_per_thread;
        const std::size_t num_blocks = (num_threads+block_dim-1)/block_dim;
        pack_kernel<LMap><<<num_blocks,block_dim,0,*stream_ptr>>>(pack_is, elements_per_thread);
    }

    template<typename UnPackIterationSpace>
    static void unpack(UnPackIterationSpace&& unpack_is, void* arg) {
        using coordinate_type = typename UnPackIterationSpace::coordinate_t;
        auto stream_ptr = reinterpret_cast<cudaStream_t*>(arg);
        const std::size_t num_threads = (unpack_is.m_buffer_desc.m_size + elements_per_thread-1)
            /elements_per_thread;
        const std::size_t num_blocks = (num_threads+block_dim-1)/block_dim;
        unpack_kernel<LMap><<<num_blocks,block_dim,0,*stream_ptr>>>(unpack_is, elements_per_thread);
    }
};
#endif

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_PACK_KERNELS_HPP */
