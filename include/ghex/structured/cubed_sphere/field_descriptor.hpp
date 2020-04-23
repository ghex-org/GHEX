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
#include <gridtools/common/array.hpp>
#include <gridtools/common/layout_map.hpp>
#include "../pack_kernels.hpp"
#include "./halo_generator.hpp"

namespace gridtools {
namespace ghex {
namespace structured {
namespace cubed_sphere {

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
        unpack_iteration_space(BufferDesc&& bd, IS&& is, const transform& t, int c_, bool is_vector)
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
        std::array<int,3>{dom_.first()[1],dom_.first()[2],dom_.first()[3]},
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
                (int)base::m_num_components};
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
                (int)base::m_num_components};
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

} // namespace cubed_sphere
} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_FIELD_DESCRIPTOR_HPP
