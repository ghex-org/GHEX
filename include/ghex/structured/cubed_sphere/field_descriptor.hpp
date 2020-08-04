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
#include "../field_descriptor.hpp"
#include "./halo_generator.hpp"

namespace gridtools {
namespace ghex {
namespace structured {
namespace cubed_sphere {

template<typename T, typename Arch, int X=3, int Y=2, int Z=1, int C=0>
class field_descriptor : public structured::field_descriptor<T,Arch,domain_descriptor,X,Y,Z,C>
{
public: // member types
    using base = structured::field_descriptor<T,Arch,domain_descriptor,X,Y,Z,C>;

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
    using serialization_type       = serialization<arch_type,layout_map>;
    using pack_iteration_space     = typename base::pack_iteration_space;
    
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
    };

    int             m_edge_size;             /// cube size

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
        offsets_,
        extents_,
        num_components_,
        is_vector_field_,
        d_id_)
    , m_edge_size{dom_.edge_size()}
    {}
    field_descriptor(field_descriptor&&) noexcept = default;
    field_descriptor(const field_descriptor&) noexcept = default;
    field_descriptor& operator=(field_descriptor&&) noexcept = default;
    field_descriptor& operator=(const field_descriptor&) noexcept = default;
    
    /** @brief returns the size of the cube (number of cells along an edge) */
    int edge_size() const noexcept { return m_edge_size; }

    template<typename IndexContainer>
    void pack(T* buffer, const IndexContainer& c, void* arg) {
        // loop over pattern's iteration spaces
        for (const auto& is : c) {
            const size_type size = is.size()*base::num_components();
            serialization_type::pack ( make_pack_is(is, buffer, size), arg );
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
            // number of values to unpack
            const size_type size = is.size()*base::num_components();
            serialization_type::unpack ( make_unpack_is(is, buffer, size, *t), arg );
            buffer += size;
        }
    }
private: // implementation
    template<typename IterationSpace>
    pack_iteration_space make_pack_is(const IterationSpace& is, T* buffer, size_type size) {
        return {make_buffer_desc<typename base::template buffer_descriptor<T*>>(is,buffer,size),
                make_is<typename base::template basic_iteration_space<const T*>>(is)};
    }

    template<typename IterationSpace>
    unpack_iteration_space make_unpack_is(const IterationSpace& is, const T* buffer, size_type size, 
        const transform& t) {
        return {make_buffer_desc<typename base::template buffer_descriptor<const T*>>(is,buffer,size),
                make_is<typename base::template basic_iteration_space<T*>>(is),
                t, m_edge_size, base::m_is_vector_field};
    }

    template<typename BufferDesc, typename IterationSpace, typename Buffer>
    BufferDesc make_buffer_desc(const IterationSpace& is, Buffer buffer, size_type size) {
        // description of the halo in the buffer
        coordinate_type buffer_offset;
        std::copy(is.global().first().begin()+1, is.global().first().end(), buffer_offset.begin()); 
        if (has_components::value)
            buffer_offset[dimension::value-1] = 0;
        coordinate_type buffer_extents;
        std::copy(is.global().last().begin()+1, is.global().last().end(), buffer_extents.begin()); 
        if (has_components::value)
            buffer_extents[dimension::value-1] = base::m_num_components;
        buffer_extents = buffer_extents - buffer_offset+1;
        strides_type buffer_strides;
        ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
            apply<layout_map,value_type>(buffer_extents,buffer_strides,0u);
        return {buffer, buffer_offset, buffer_strides, size};
    }

    template<typename IS, typename IterationSpace>
    IS make_is(const IterationSpace& is) {
        // description of the halo in the local domain
        coordinate_type data_first;
        coordinate_type data_last;
        std::copy(is.local().first().begin()+1, is.local().first().end(), data_first.begin());
        std::copy(is.local().last().begin()+1, is.local().last().end(), data_last.begin());
        if (has_components::value) {
            data_first[dimension::value-1] = 0;
            data_last[dimension::value-1] = base::m_num_components-1;
        }
        const coordinate_type local_extents = data_last-data_first+1;
        strides_type local_strides;
        ::gridtools::ghex::structured::detail::compute_strides<dimension::value>::template
            apply<layout_map>(local_extents,local_strides);
        return {base::m_data, base::m_dom_first, base::m_offsets,
                data_first, data_last, base::m_byte_strides, local_strides};
    }
};

} // namespace cubed_sphere
} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_STRUCTURED_CUBED_SPHERE_FIELD_DESCRIPTOR_HPP
