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
#ifndef INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_FIELD_DESCRIPTOR_HPP
#define INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_FIELD_DESCRIPTOR_HPP

#include <vector>
#include <algorithm>
#include "./halo_generator.hpp"
#include "../field_utils.hpp"
#include "../../common/utils.hpp"
#include "../../arch_traits.hpp"
#include <gridtools/common/array.hpp>
#include <gridtools/common/layout_map.hpp>

namespace gridtools {
    namespace ghex {
        namespace structured {
            namespace cubed_sphere {

                /** @brief Helper class to dispatch to CPU/GPU implementations of pack/unpack kernels
                  * @tparam Arch Architecture type
                  * @tparam LayoutMap Data layout map*/
                template<typename Arch, typename LayoutMap>
                struct serialization;
                
                template<typename LayoutMap>
                struct serialization<::gridtools::ghex::cpu,LayoutMap> {
                    template<typename PackIterationSpace>
                    static void pack(PackIterationSpace&& pack_is, void*) {
                        using coordinate_type = typename PackIterationSpace::coordinate_t;
                        ::gridtools::ghex::detail::for_loop<4,4,LayoutMap>::template apply(
                            [&pack_is](int x, int y, int z, int c) {
                                pack_is.buffer(coordinate_type{x,y,z,c}) =
                                pack_is.data(coordinate_type{x,y,z,c});},
                            pack_is.m_data_is.m_first,
                            pack_is.m_data_is.m_last);
                    }

                    template<typename UnPackIterationSpace>
                    static void unpack(UnPackIterationSpace&& unpack_is, void*) {
                        using coordinate_type = typename UnPackIterationSpace::coordinate_t;
                        ::gridtools::ghex::detail::for_loop<4,4,LayoutMap>::template apply(
                            [&unpack_is](int x, int y, int z, int c) {
                                unpack_is.data(coordinate_type{x,y,z,c}) =
                                unpack_is.buffer(coordinate_type{x,y,z,c});
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

                template<typename T, typename Arch, int...Order>
                using base_fd_t =
                    ::gridtools::ghex::structured::field_descriptor<T,Arch,domain_descriptor,Order...>;

                template<typename T, typename Arch, int X=3, int Y=2, int Z=1, int C=0>
                class field_descriptor : public base_fd_t<T,Arch,X,Y,Z,C> {
                public: // member types
                    using base = base_fd_t<T,Arch,X,Y,Z,C>;

                    using value_type               = typename base::value_type;
                    using arch_type                = typename base::arch_type;
                    using device_id_type           = typename base::device_id_type;
                    using domain_descriptor_type   = typename base::domain_descriptor_type;
                    using dimension                = typename base::dimension;
                    using layout_map               = typename base::layout_map;
                    using has_components           = typename base::has_components;
                    using domain_id_type           = typename base::domain_id_type;
                    using coordinate_type          = typename base::coordinate_type;
                    using size_type                = typename base::size_type;
                    using strides_type             = typename base::strides_type;
                    
                    using serialization_type     = serialization<arch_type,layout_map>;
                    
                    struct pack_iteration_space : public base::pack_iteration_space {
                        using base_is = typename base::pack_iteration_space;
                        
                        template<typename BufferDesc, typename IS>
                        pack_iteration_space(BufferDesc&& bd, IS&& is)
                        : base_is{std::forward<BufferDesc>(bd), std::forward<IS>(is)}
                        {}

                        pack_iteration_space(pack_iteration_space&&) noexcept = default;
                        pack_iteration_space(const pack_iteration_space&) noexcept = default;
                        pack_iteration_space& operator=(pack_iteration_space&&) noexcept = default;
                        pack_iteration_space& operator=(const pack_iteration_space&) noexcept = default;

                        /** @brief accesses buffer at specified local coordinate
                          * @param coord 4-dimensional array (x, y, z, component) in local coordinate system
                          * @return reference to the value in the buffer */
                        GT_FUNCTION
                        T& buffer(const coordinate_type& coord) const noexcept {
                            // make global coordinates
                            const coordinate_type global_coord{
                                coord[0] + base_is::m_data_is.m_domain_first[0],
                                coord[1] + base_is::m_data_is.m_domain_first[1],
                                coord[2] + base_is::m_data_is.m_domain_first[2],
                                coord[3]};
                            // compute buffer coordinates, relative to the buffer origin
                            const coordinate_type buffer_coord = global_coord - base_is::m_buffer_desc.m_first;
                            // dot product with strides to compute address
                            const auto memory_location =
                                base_is::m_buffer_desc.m_strides[0]*buffer_coord[0] +
                                base_is::m_buffer_desc.m_strides[1]*buffer_coord[1] +
                                base_is::m_buffer_desc.m_strides[2]*buffer_coord[2] +
                                base_is::m_buffer_desc.m_strides[3]*buffer_coord[3];
                            return *reinterpret_cast<T*>(
                                reinterpret_cast<char*>(base_is::m_buffer_desc.m_ptr) + memory_location);
                        }

                        /** @brief accesses field at specified local coordinate
                          * @param coord 4-dimensional array (x, y, z, component) in local coordinate system
                          * @return const reference to the value in the field */
                        GT_FUNCTION
                        const T& data(const coordinate_type& coord) const noexcept {
                            // make data memory coordinates from local coordinates
                            const coordinate_type data_coord = coord + base_is::m_data_is.m_offset;
                            // dot product with strides to compute address
                            const auto memory_location =
                                base_is::m_data_is.m_strides[0]*data_coord[0] +
                                base_is::m_data_is.m_strides[1]*data_coord[1] +
                                base_is::m_data_is.m_strides[2]*data_coord[2] +
                                base_is::m_data_is.m_strides[3]*data_coord[3];
                            return *reinterpret_cast<const T*>(
                                reinterpret_cast<const char*>(base_is::m_data_is.m_ptr) + memory_location);
                        }
                    };

                    struct unpack_iteration_space : public base::unpack_iteration_space {
                        using base_is = typename base::unpack_iteration_space;

                        template<typename BufferDesc, typename IS>
                        unpack_iteration_space(BufferDesc&& bd, IS&& is, const transform& t,
                            int c_, bool is_vector)
                        : base_is{std::forward<BufferDesc>(bd), std::forward<IS>(is)}
                        , m_transform{t}
                        , c{c_}
                        , m_is_vector{is_vector}
                        {}
    
                        unpack_iteration_space(unpack_iteration_space&&) noexcept = default;
                        unpack_iteration_space(const unpack_iteration_space&) noexcept = default;
                        unpack_iteration_space& operator=(unpack_iteration_space&&) noexcept = default;
                        unpack_iteration_space& operator=(const unpack_iteration_space&) noexcept = default;
                                    
                        const transform m_transform;
                        const int c;
                        const bool m_is_vector;

                        /** @brief accesses buffer at specified local coordinate
                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
                          * @return value in the buffer */
                        GT_FUNCTION
                        T buffer(const coordinate_type& coord) const noexcept {
                            // make global coordinates
                            const auto x = coord[0] + base_is::m_data_is.m_domain_first[0];
                            const auto y = coord[1] + base_is::m_data_is.m_domain_first[1];
                            const auto z = coord[2] + base_is::m_data_is.m_domain_first[2];
                            // transform to the neighbor's coordinate system
                            const auto xy = m_transform(x,y,c);
                            const coordinate_type tile_coord{xy[0],xy[1],z,coord[3]};
                            // compute buffer coordinates, relative to the buffer origin
                            const coordinate_type buffer_coord = tile_coord - base_is::m_buffer_desc.m_first;
                            // dot product with strides to compute address
                            const auto memory_location =
                                base_is::m_buffer_desc.m_strides[0]*buffer_coord[0] +
                                base_is::m_buffer_desc.m_strides[1]*buffer_coord[1] +
                                base_is::m_buffer_desc.m_strides[2]*buffer_coord[2] +
                                base_is::m_buffer_desc.m_strides[3]*buffer_coord[3];
                            // reverse the x component if necessary
                            if (m_is_vector && coord[3]==0 && m_transform.m_rotation[2]==-1)
                                return -(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(
                                    base_is::m_buffer_desc.m_ptr) + memory_location));
                            // reverse the y component if necessary
                            else if (m_is_vector && coord[3]==1 && m_transform.m_rotation[1]==-1)
                                return -(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(
                                    base_is::m_buffer_desc.m_ptr) + memory_location));
                            else
                                return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(
                                    base_is::m_buffer_desc.m_ptr) + memory_location);
                        }

                        /** @brief accesses field at specified local coordinate
                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
                          * @return reference to the value in the field */
                        GT_FUNCTION
                        T& data(const coordinate_type& coord) const noexcept {
                            // make data memory coordinates from local coordinates
                            const coordinate_type data_coord = coord + base_is::m_data_is.m_offset;
                            // dot product with strides to compute address
                            const auto memory_location =
                                base_is::m_data_is.m_strides[0]*data_coord[0] +
                                base_is::m_data_is.m_strides[1]*data_coord[1] +
                                base_is::m_data_is.m_strides[2]*data_coord[2] +
                                base_is::m_data_is.m_strides[3]*data_coord[3];
                            return *reinterpret_cast<T*>(
                                reinterpret_cast<char*>(base_is::m_data_is.m_ptr) + memory_location);
                        }
                    };

                    int             m_c;             /// cube size

                    template<typename Array>
                    field_descriptor(
                            const domain_descriptor& dom_,
                            value_type* data_,
                            const Array& offsets_,
                            const Array& extents_,
                            unsigned int num_components_ = 1u,
                            bool is_vector_field_ = false,
                            device_id_type d_id_ = 0)
                    : base(
                        dom_,
                        data_,
                        coordinate_type{offsets_[0], offsets_[1], offsets_[2], 0},
                        coordinate_type{extents_[0], extents_[1], extents_[2], (int)num_components_},
                        num_components_,
                        is_vector_field_,
                        d_id_)
                    , m_c{dom_.x()}
                    {}
                    field_descriptor(field_descriptor&&) noexcept = default;
                    field_descriptor(const field_descriptor&) noexcept = default;
                    field_descriptor& operator=(field_descriptor&&) noexcept = default;
                    field_descriptor& operator=(const field_descriptor&) noexcept = default;
                    
                    /** @brief returns the size of the cube (number of cells along an edge) */
                    int x() const noexcept { return m_c; }

                    template<typename IndexContainer>
                    void pack(T* buffer, const IndexContainer& c, void* arg) {
                        // loop over pattern's iteration spaces
                        for (const auto& is : c) {
                            // 4-D description of the halo in the buffer
                            const coordinate_type buffer_offset {
                                is.global().first()[1],
                                is.global().first()[2],
                                is.global().first()[3],
                                0};
                            const coordinate_type buffer_extents {
                                is.global().last()[1]-is.global().first()[1]+1,
                                is.global().last()[2]-is.global().first()[2]+1,
                                is.global().last()[3]-is.global().first()[3]+1,
                                base::m_num_components};
                            strides_type buffer_strides;
                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
                                apply<layout_map,value_type>(buffer_extents,buffer_strides,0u);
                            // 4-D description of the halo in the local domain
                            coordinate_type data_first;
                            coordinate_type data_last;
                            data_first[3] = 0;
                            data_last[3] = base::m_num_components-1;
                            std::copy(is.local().first().begin()+1, is.local().first().end(), data_first.begin());
                            std::copy(is.local().last().begin()+1, is.local().last().end(), data_last.begin());
                            const coordinate_type local_extents{
                                data_last[0]-data_first[0]+1,
                                data_last[1]-data_first[1]+1,
                                data_last[2]-data_first[2]+1,
                                data_last[3]-data_first[3]+1};
                            strides_type local_strides;
                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
                                apply<layout_map>(local_extents,local_strides);
                            // number of values to pack
                            const size_type size = is.size()*base::num_components();
                            // dispatch to specialized packer
                            serialization_type::pack(
                                pack_iteration_space{
                                    typename base::template buffer_descriptor<T*>{
                                        buffer,
                                        buffer_offset,
                                        buffer_strides,
                                        size},
                                    typename base::template basic_iteration_space<const T*>{
                                        base::m_data,
                                        base::m_dom_first,
                                        base::m_offsets,
                                        data_first,
                                        data_last,
                                        base::m_byte_strides,
                                        local_strides}},
                                arg);
                            buffer += size;
                        }
                    }

                    template<typename IndexContainer>
                    void unpack(const T* buffer, const IndexContainer& c, void* arg) {
                        // loop over pattern's iteration spaces
                        for (const auto& is : c) {
                            // pointer to transform matrix
                            const transform * t;
                            // check if iteration space is on different tile
                            if (is.global().first()[0] != base::m_dom.domain_id().tile) {
                                // find neighbor tile's direction: -x,+x,-y,+y
                                int n;
                                for (n=0; n<4; ++n)
                                    if (tile_lu[base::m_dom.domain_id().tile][n] == is.global().first()[0])
                                        break;
                                // assign transform
                                t = &transform_lu[base::m_dom.domain_id().tile][n];
                            }
                            else
                                t = &identity_transform;
                            // 4-D description of the halo in the buffer
                            const coordinate_type buffer_offset {
                                is.global().first()[1],
                                is.global().first()[2],
                                is.global().first()[3],
                                0};
                            const coordinate_type buffer_extents {
                                is.global().last()[1]-is.global().first()[1]+1,
                                is.global().last()[2]-is.global().first()[2]+1,
                                is.global().last()[3]-is.global().first()[3]+1,
                                base::m_num_components};
                            strides_type buffer_strides;
                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
                                apply<layout_map,value_type>(buffer_extents,buffer_strides,0u);
                            // 4-D description of the halo in the local domain
                            coordinate_type data_first;
                            coordinate_type data_last;
                            data_first[3] = 0;
                            data_last[3] = base::m_num_components-1;
                            std::copy(is.local().first().begin()+1, is.local().first().end(), data_first.begin());
                            std::copy(is.local().last().begin()+1, is.local().last().end(), data_last.begin());
                            const coordinate_type local_extents{
                                data_last[0]-data_first[0]+1,
                                data_last[1]-data_first[1]+1,
                                data_last[2]-data_first[2]+1,
                                data_last[3]-data_first[3]+1};
                            strides_type local_strides;
                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
                                apply<layout_map>(local_extents,local_strides);
                            // number of values to unpack
                            const size_type size = is.size()*base::num_components();
                            // dispatch to specialized unpacker
                            serialization_type::unpack(
                                unpack_iteration_space{
                                    typename base::template buffer_descriptor<const T*>{
                                        buffer,
                                        buffer_offset,
                                        buffer_strides,
                                        size},
                                    typename base::template basic_iteration_space<T*>{
                                        base::m_data,
                                        base::m_dom_first,
                                        base::m_offsets,
                                        data_first,
                                        data_last,
                                        base::m_byte_strides,
                                        local_strides},
                                    *t,
                                    m_c,
                                    base::m_is_vector_field},
                                arg);
                            buffer += size;
                        }
                    }
                };

//                /** @brief Describes a field spanning a subdomain of a cubed-sphere tile
//                  * @tparam T value type
//                  * @tparam Arch Architecture type
//                  * @tparam ComponentOrder Data layout order parameter (3 == stride 1, 0 == largest stride)
//                  * @tparam XOrder Data layout order parameter (3 == stride 1, 0 == largest stride)
//                  * @tparam YOrder Data layout order parameter (3 == stride 1, 0 == largest stride)
//                  * @tparam ZOrder Data layout order parameter (3 == stride 1, 0 == largest stride)*/
//                template<typename T, typename Arch, int ComponentOrder=0, int XOrder=3, int YOrder=2, int ZOrder=1>
//                class field_descriptor {
//                public: // member types
//                    using value_type             = T;
//                    using arch_type              = Arch;
//                    using device_id_type         = typename arch_traits<arch_type>::device_id_type;
//                    using domain_descriptor_type = domain_descriptor;
//                    using dimension              = std::integral_constant<int,4>;
//                    using layout_map             = ::gridtools::layout_map<ComponentOrder,XOrder,YOrder,ZOrder>;
//                    using domain_id_type         = domain_id_t;
//                    using coordinate_type        = ::gridtools::array<
//                        typename domain_descriptor_type::coordinate_type::value_type, dimension::value>;
//                    using size_type              = unsigned int;
//                    using strides_type           = ::gridtools::array<size_type, dimension::value>;
//                    using serialization_type     = serialization<arch_type,layout_map>;
//
//                    // holds buffer information (read-write access)
//                    struct buffer_descriptor {
//                        T* m_ptr;
//                        const coordinate_type m_first;
//                        const strides_type m_strides;
//                        const size_type m_size;
//                    };
//                    
//                    // holds buffer information (read only access)
//                    struct const_buffer_descriptor {
//                        const T* m_ptr;
//                        const coordinate_type m_first;
//                        const strides_type m_strides;
//                        const size_type m_size;
//                    };
//                    
//                    // holds halo iteration space information (read-write access)
//                    struct basic_iteration_space {
//                        T* m_ptr;
//                        const coordinate_type m_domain_first;
//                        const coordinate_type m_offset;
//                        const coordinate_type m_first;
//                        const coordinate_type m_last;
//                        const strides_type m_strides;
//                        const strides_type m_local_strides;
//                    };
//
//                    // holds halo iteration space information (read only access)
//                    struct const_basic_iteration_space {
//                        const T* m_ptr;
//                        const coordinate_type m_domain_first;
//                        const coordinate_type m_offset;
//                        const coordinate_type m_first;
//                        const coordinate_type m_last;
//                        const strides_type m_strides;
//                        const strides_type m_local_strides;
//                    };
//
//                    // iteration space for packing data
//                    // does not apply any transform (data is packed according to the local coordinate system)
//                    struct pack_iteration_space {
//                        using value_t = T;
//                        using coordinate_t = coordinate_type;
//                        const buffer_descriptor m_buffer_desc;
//                        const const_basic_iteration_space m_data_is;
//                        
//                        /** @brief accesses buffer at specified local coordinate
//                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
//                          * @return reference to the value in the buffer */
//                        GT_FUNCTION
//                        T& buffer(const coordinate_type& coord) const noexcept {
//                            // make global coordinates
//                            const coordinate_type global_coord{
//                                coord[0],
//                                coord[1] + m_data_is.m_domain_first[1],
//                                coord[2] + m_data_is.m_domain_first[2],
//                                coord[3]};
//                            // compute buffer coordinates, relative to the buffer origin
//                            const coordinate_type buffer_coord = global_coord - m_buffer_desc.m_first;
//                            // dot product with strides to compute address
//                            const auto memory_location =
//                                m_buffer_desc.m_strides[0]*buffer_coord[0] +
//                                m_buffer_desc.m_strides[1]*buffer_coord[1] +
//                                m_buffer_desc.m_strides[2]*buffer_coord[2] +
//                                m_buffer_desc.m_strides[3]*buffer_coord[3];
//                            return *reinterpret_cast<T*>(
//                                reinterpret_cast<char*>(m_buffer_desc.m_ptr) + memory_location);
//                        }
//
//                        /** @brief accesses field at specified local coordinate
//                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
//                          * @return const reference to the value in the field */
//                        GT_FUNCTION
//                        const T& data(const coordinate_type& coord) const noexcept {
//                            // make data memory coordinates from local coordinates
//                            const coordinate_type data_coord = coord + m_data_is.m_offset;
//                            // dot product with strides to compute address
//                            const auto memory_location =
//                                m_data_is.m_strides[0]*data_coord[0] +
//                                m_data_is.m_strides[1]*data_coord[1] +
//                                m_data_is.m_strides[2]*data_coord[2] +
//                                m_data_is.m_strides[3]*data_coord[3];
//                            return *reinterpret_cast<const T*>(
//                                reinterpret_cast<const char*>(m_data_is.m_ptr) + memory_location);
//                        }
//                    };
//
//                    // iteration space for unpacking data
//                    // transforms coordinates and values depending on the transformation matrix
//                    struct unpack_iteration_space {
//                        using value_t = T;
//                        using coordinate_t = coordinate_type;
//                        const const_buffer_descriptor m_buffer_desc;
//                        const basic_iteration_space m_data_is;
//                        const transform m_transform;
//                        const int c;
//                        const bool m_is_vector;
//
//                        /** @brief accesses buffer at specified local coordinate
//                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
//                          * @return value in the buffer */
//                        GT_FUNCTION
//                        T buffer(const coordinate_type& coord) const noexcept {
//                            // make global coordinates
//                            const auto x = coord[1] + m_data_is.m_domain_first[1];
//                            const auto y = coord[2] + m_data_is.m_domain_first[2];
//                            // transform to the neighbor's coordinate system
//                            const auto xy = m_transform(x,y,c);
//                            const coordinate_type tile_coord{coord[0],xy[0],xy[1],coord[3]};
//                            // compute buffer coordinates, relative to the buffer origin
//                            const coordinate_type buffer_coord = tile_coord - m_buffer_desc.m_first;
//                            // dot product with strides to compute address
//                            const auto memory_location =
//                                m_buffer_desc.m_strides[0]*buffer_coord[0] +
//                                m_buffer_desc.m_strides[1]*buffer_coord[1] +
//                                m_buffer_desc.m_strides[2]*buffer_coord[2] +
//                                m_buffer_desc.m_strides[3]*buffer_coord[3];
//                            // reverse the x component if necessary
//                            if (m_is_vector && coord[0]==0 && m_transform.m_rotation[2]==-1)
//                                return -(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_buffer_desc.m_ptr) +
//                                    memory_location));
//                            // reverse the y component if necessary
//                            else if (m_is_vector && coord[0]==1 && m_transform.m_rotation[1]==-1)
//                                return -(*reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_buffer_desc.m_ptr) +
//                                    memory_location));
//                            else
//                                return *reinterpret_cast<const T*>(reinterpret_cast<const char*>(m_buffer_desc.m_ptr) +
//                                    memory_location);
//                        }
//
//                        /** @brief accesses field at specified local coordinate
//                          * @param coord 4-dimensional array (component, x, y, z) in local coordinate system
//                          * @return reference to the value in the field */
//                        GT_FUNCTION
//                        T& data(const coordinate_type& coord) const noexcept {
//                            // make data memory coordinates from local coordinates
//                            const coordinate_type data_coord = coord + m_data_is.m_offset;
//                            // dot product with strides to compute address
//                            const auto memory_location =
//                                m_data_is.m_strides[0]*data_coord[0] +
//                                m_data_is.m_strides[1]*data_coord[1] +
//                                m_data_is.m_strides[2]*data_coord[2] +
//                                m_data_is.m_strides[3]*data_coord[3];
//                            return *reinterpret_cast<T*>(
//                                reinterpret_cast<char*>(m_data_is.m_ptr) + memory_location);
//                        }
//                    };
//
//                private: // members
//                    domain_id_type  m_dom_id;        /// domain id
//                    int             m_c;             /// cube size
//                    bool            m_is_vector;     /// is this data describing a vector field 
//                    value_type*     m_data;          /// pointer to data
//                    coordinate_type m_dom_first;     /// global coordinate of first domain cell
//                    coordinate_type m_offsets;       /// offset from beginning of memory to the first domain cell
//                    coordinate_type m_extents;       /// extent of memory (including halos)
//                    device_id_type  m_device_id;     /// device id
//                    strides_type    m_byte_strides;  /// memory strides in bytes
//
//                public: // ctors
//                    /** @brief construct field_descriptor from existing data
//                      * @tparam Array 3-dimensional array-like type
//                      * @param dom domain descriptor
//                      * @param data pointer to data
//                      * @param offsets (x,y,z)-offset of first phyisical value (not halo) from the orign of the memory
//                      * @param extents (x,y,z)-extent of the memory (including buffer regions)
//                      * @param n_components number of components in this field
//                      * @param is_vector whether this field is a vector or a collection of scalars
//                      * @param d_id device id */
//                    template<typename Array>
//                    field_descriptor(const domain_descriptor& dom, value_type* data,
//                                     const Array& offsets, const Array& extents, unsigned int n_components,
//                                     bool is_vector_ = false, device_id_type d_id = 0)
//                    : m_dom_id{dom.domain_id()}
//                    , m_c{dom.x()}
//                    , m_is_vector{is_vector_}
//                    , m_data{data}
//                    , m_device_id{d_id}
//                    { 
//                        std::copy(dom.first().begin(), dom.first().end(), m_dom_first.begin());
//                        m_offsets[0] = 0;
//                        std::copy(offsets.begin(), offsets.end(), m_offsets.begin()+1);
//                        m_extents[0] = n_components;
//                        std::copy(extents.begin(), extents.end(), m_extents.begin()+1);
//                        // compute strides
//                        ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
//                            apply<layout_map,value_type>(m_extents,m_byte_strides,0u);
//                        m_dom_first[0] = 0;
//                    }
//
//                    field_descriptor(field_descriptor&&) noexcept = default;
//                    field_descriptor(const field_descriptor&) noexcept = default;
//                    field_descriptor& operator=(field_descriptor&&) noexcept = default;
//                    field_descriptor& operator=(const field_descriptor&) noexcept = default;
//
//                public: // member functions
//            
//                    /** @brief returns the device id */
//                    typename arch_traits<arch_type>::device_id_type device_id() const { return m_device_id; }
//                    /** @brief returns the domain id */
//                    domain_id_type domain_id() const noexcept { return m_dom_id; }
//                    /** @brief returns the field 4-D extents (c,x,y,z) */
//                    const coordinate_type& extents() const noexcept { return m_extents; }
//                    /** @brief returns the field 4-D offsets (c,x,y,z) */
//                    const coordinate_type& offsets() const noexcept { return m_offsets; }
//                    /** @brief returns the field 4-D byte strides (c,x,y,z) */
//                    const strides_type& byte_strides() const noexcept { return m_byte_strides; }
//                    /** @brief returns the pointer to the data */
//                    value_type* data() const { return m_data; }
//                    /** @brief set a new pointer to the data */
//                    void set_data(value_type* ptr) { m_data = ptr; }
//                    /** @brief returns the number of components c */
//                    int num_components() const noexcept { return m_extents[0]; }
//                    /** @brief returns the size of the cube (number of cells along an edge) */
//                    int x() const noexcept { return m_c; }
//                    /** @brief returns true if this field describes a vector field */
//                    bool is_vector() const noexcept { return m_is_vector; }
//                    
//                    template<typename IndexContainer>
//                    void pack(T* buffer, const IndexContainer& c, void* arg) {
//                        // loop over pattern's iteration spaces
//                        for (const auto& is : c) {
//                            // 4-D description of the halo in the buffer
//                            const coordinate_type buffer_offset {
//                                0,
//                                is.global().first()[1],
//                                is.global().first()[2],
//                                is.global().first()[3]};
//                            const coordinate_type buffer_extents {
//                                m_extents[0],
//                                is.global().last()[1]-is.global().first()[1]+1,
//                                is.global().last()[2]-is.global().first()[2]+1,
//                                is.global().last()[3]-is.global().first()[3]+1};
//                            strides_type buffer_strides;
//                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
//                                apply<layout_map,value_type>(buffer_extents,buffer_strides,0u);
//                            // 4-D description of the halo in the local domain
//                            coordinate_type data_first;
//                            coordinate_type data_last;
//                            data_first[0] = 0;
//                            data_last[0] = m_extents[0]-1;
//                            std::copy(is.local().first().begin()+1, is.local().first().end(), data_first.begin()+1);
//                            std::copy(is.local().last().begin()+1, is.local().last().end(), data_last.begin()+1);
//                            const coordinate_type local_extents{
//                                data_last[0]-data_first[0]+1,
//                                data_last[1]-data_first[1]+1,
//                                data_last[2]-data_first[2]+1,
//                                data_last[3]-data_first[3]+1};
//                            strides_type local_strides;
//                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
//                                apply<layout_map>(local_extents,local_strides);
//                            // number of values to pack
//                            const size_type size = is.size()*num_components();
//                            // dispatch to specialized packer
//                            serialization_type::pack(
//                                pack_iteration_space{
//                                    buffer_descriptor{
//                                        buffer,
//                                        buffer_offset,
//                                        buffer_strides,
//                                        size},
//                                    const_basic_iteration_space{
//                                        m_data,
//                                        m_dom_first,
//                                        m_offsets,
//                                        data_first,
//                                        data_last,
//                                        m_byte_strides,
//                                        local_strides}},
//                                arg);
//                            buffer += size;
//                        }
//                    }
//            
//                    template<typename IndexContainer>
//                    void unpack(const T* buffer, const IndexContainer& c, void* arg) {
//                        // loop over pattern's iteration spaces
//                        for (const auto& is : c) {
//                            // pointer to transform matrix
//                            const transform * t;
//                            // check if iteration space is on different tile
//                            if (is.global().first()[0] != m_dom_id.tile) {
//                                // find neighbor tile's direction: -x,+x,-y,+y
//                                int n;
//                                for (n=0; n<4; ++n)
//                                    if (tile_lu[m_dom_id.tile][n] == is.global().first()[0])
//                                        break;
//                                // assign transform
//                                t = &transform_lu[m_dom_id.tile][n];
//                            }
//                            else
//                                t = &identity_transform;
//                            // 4-D description of the halo in the buffer
//                            const coordinate_type buffer_offset {
//                                0,
//                                is.global().first()[1],
//                                is.global().first()[2],
//                                is.global().first()[3]};
//                            const coordinate_type buffer_extents {
//                                m_extents[0],
//                                is.global().last()[1]-is.global().first()[1]+1,
//                                is.global().last()[2]-is.global().first()[2]+1,
//                                is.global().last()[3]-is.global().first()[3]+1};
//                            strides_type buffer_strides;
//                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
//                                apply<layout_map,value_type>(buffer_extents,buffer_strides,0u);
//                            // 4-D description of the halo in the local domain
//                            coordinate_type data_first;
//                            coordinate_type data_last;
//                            data_first[0] = 0;
//                            data_last[0] = m_extents[0]-1;
//                            std::copy(is.local().first().begin()+1, is.local().first().end(), data_first.begin()+1);
//                            std::copy(is.local().last().begin()+1, is.local().last().end(), data_last.begin()+1);
//                            const coordinate_type local_extents{
//                                data_last[0]-data_first[0]+1,
//                                data_last[1]-data_first[1]+1,
//                                data_last[2]-data_first[2]+1,
//                                data_last[3]-data_first[3]+1};
//                            strides_type local_strides;
//                            ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
//                                apply<layout_map>(local_extents,local_strides);
//                            // number of values to unpack
//                            const size_type size = is.size()*num_components();
//                            // dispatch to specialized unpacker
//                            serialization_type::unpack(
//                                unpack_iteration_space{
//                                    const_buffer_descriptor{
//                                        buffer,
//                                        buffer_offset,
//                                        buffer_strides,
//                                        size},
//                                    basic_iteration_space{
//                                        m_data,
//                                        m_dom_first,
//                                        m_offsets,
//                                        data_first,
//                                        data_last,
//                                        m_byte_strides,
//                                        local_strides},
//                                    *t,
//                                    m_c,
//                                    m_is_vector},
//                                arg);
//                            buffer += size;
//                        }
//                    }
//                };

            } // namespace cubed_sphere
        } // namespace structured
    } // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_FIELD_DESCRIPTOR_HPP
