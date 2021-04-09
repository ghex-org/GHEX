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

#include <cstring>
#include "./field_utils.hpp"
#include "../common/utils.hpp"
#include "../arch_traits.hpp"

#include "../common/defs.hpp"
#ifdef GHEX_CUDACC
#include "../common/cuda_runtime.hpp"
#endif

namespace gridtools {
namespace ghex {
namespace structured {

// this is to exclude index Idx from a layout map
// and return a reduced layout map
template<std::size_t Idx, typename Seq, typename LMap>
struct reduced_layout_map;
template<std::size_t Idx, std::size_t... Ms, typename LMap>
struct reduced_layout_map<Idx, std::index_sequence<Ms...>, LMap>{
    using type = gridtools::layout_map<LMap::at(Ms<Idx ? Ms : Ms+1)...>;
};

/** @brief Helper class to dispatch to CPU/GPU implementations of pack/unpack kernels
  * @tparam Arch Architecture type
  * @tparam LayoutMap Data layout map*/
template<typename Arch, typename LayoutMap>
struct serialization {
    template<typename PackIterationSpace>
    static void pack(PackIterationSpace&& pack_is, void*) {
        using coordinate_type = typename PackIterationSpace::coordinate_t;
        static constexpr auto D = coordinate_type::size();
        ::gridtools::ghex::detail::for_loop<D,D,LayoutMap>::template apply(
            [&pack_is](auto... xs) {
                pack_is.buffer(coordinate_type{xs...}) = pack_is.data(coordinate_type{xs...});
            },
            pack_is.m_data_is.m_first,
            pack_is.m_data_is.m_last);
    }

    template<typename UnPackIterationSpace>
    static void unpack(UnPackIterationSpace&& unpack_is, void*) {
        using coordinate_type = typename UnPackIterationSpace::coordinate_t;
        static constexpr auto D = coordinate_type::size();
        ::gridtools::ghex::detail::for_loop<D,D,LayoutMap>::template apply(
            [&unpack_is](auto... xs) {
                unpack_is.data(coordinate_type{xs...}) = unpack_is.buffer(coordinate_type{xs...});
            },
            unpack_is.m_data_is.m_first,
            unpack_is.m_data_is.m_last);
    }

    template<typename PackIterationSpace>
    static void pack_batch(PackIterationSpace&& pack_is, void*) {
        using coordinate_type = typename PackIterationSpace::coordinate_t;
        using value_type = typename PackIterationSpace::value_t;
        constexpr auto D = coordinate_type::size();
        constexpr auto cont_idx = LayoutMap::find(D-1);
        const auto x_first = pack_is.m_data_is.m_first[cont_idx];
        const auto x_last = pack_is.m_data_is.m_last[cont_idx];
        using LayoutMap2 = typename reduced_layout_map<cont_idx,std::make_index_sequence<D-1>,LayoutMap>::type;
        using scalar_coord_type = typename std::remove_cv<decltype(x_first)>::type;
        using cont_coord_type = gridtools::array<scalar_coord_type,D-1>;
        cont_coord_type first,last;
        for (std::size_t j=0, i=0; i<D; ++i) {
            if (i==cont_idx) continue;
            first[j] = pack_is.m_data_is.m_first[i];
            last[j++] = pack_is.m_data_is.m_last[i];
        }
        ::gridtools::ghex::detail::for_loop<D-1,D-1,LayoutMap2>::template apply(
            [&pack_is,&x_first,&x_last](auto... xs) {
                const cont_coord_type x0{xs...};
                coordinate_type x1;
                x1[cont_idx] = x_first;
                for (std::size_t j=0, i=0; i<D; ++i) {
                    if (i==cont_idx) continue;
                    x1[i] = x0[j++];
                }
                value_type* buffer = &(pack_is.buffer(x1));
                value_type const * field = &(pack_is.data(x1));
                std::memcpy(buffer, field, (x_last-x_first+1)*sizeof(value_type));
            },
            first,
            last);
    }

    template<typename UnPackIterationSpace>
    static void unpack_batch(UnPackIterationSpace&& unpack_is, void*) {
        using coordinate_type = typename UnPackIterationSpace::coordinate_t;
        using value_type = typename UnPackIterationSpace::value_t;
        constexpr auto D = coordinate_type::size();
        constexpr auto cont_idx = LayoutMap::find(D-1);
        const auto x_first = unpack_is.m_data_is.m_first[cont_idx];
        const auto x_last = unpack_is.m_data_is.m_last[cont_idx];
        using LayoutMap2 = typename reduced_layout_map<cont_idx,std::make_index_sequence<D-1>,LayoutMap>::type;
        using scalar_coord_type = typename std::remove_cv<decltype(x_first)>::type;
        using cont_coord_type = gridtools::array<scalar_coord_type,D-1>;
        cont_coord_type first,last;
        for (std::size_t j=0, i=0; i<D; ++i) {
            if (i==cont_idx) continue;
            first[j] = unpack_is.m_data_is.m_first[i];
            last[j++] = unpack_is.m_data_is.m_last[i];
        }
        ::gridtools::ghex::detail::for_loop<D-1,D-1,LayoutMap2>::template apply(
            [&unpack_is,&x_first,&x_last](auto... xs) {
                const cont_coord_type x0{xs...};
                coordinate_type x1;
                x1[cont_idx] = x_first;
                for (std::size_t j=0, i=0; i<D; ++i) {
                    if (i==cont_idx) continue;
                    x1[i] = x0[j++];
                }
                value_type const * buffer = &(unpack_is.buffer(x1));
                value_type * field = &(unpack_is.data(x1));
                std::memcpy(field, buffer, (x_last-x_first+1)*sizeof(value_type));
            },
            first,
            last);
    }
};

#ifdef GHEX_CUDACC
template<typename Layout, typename PackIterationSpace>
__global__ void pack_kernel(PackIterationSpace pack_is, unsigned int num_elements) {
    using value_type = typename PackIterationSpace::value_t;
    using coordinate_type = typename PackIterationSpace::coordinate_t;
    static constexpr auto D = coordinate_type::size();
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i=0; i<num_elements; ++i) {
        if (index*num_elements+i < pack_is.m_buffer_desc.m_size) {
            // compute local coordinate
            coordinate_type local_coordinate;
            ::gridtools::ghex::structured::detail::compute_coordinate<D>::template apply<Layout>(
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
    static constexpr auto D = coordinate_type::size();
    const unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
    for (unsigned int i=0; i<num_elements; ++i) {
        if (index*num_elements+i < unpack_is.m_buffer_desc.m_size) {
            // compute local coordinate
            coordinate_type local_coordinate;
            ::gridtools::ghex::structured::detail::compute_coordinate<D>::template apply<Layout>(
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
    
    template<typename PackIterationSpace>
    static void pack_batch(PackIterationSpace&& pack_is, void* arg) {
        pack(std::forward<PackIterationSpace>(pack_is), arg);
    }

    template<typename UnPackIterationSpace>
    static void unpack_batch(UnPackIterationSpace&& unpack_is, void* arg) {
        unpack(std::forward<UnPackIterationSpace>(unpack_is), arg);
    }
};
#endif

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_PACK_KERNELS_HPP */
