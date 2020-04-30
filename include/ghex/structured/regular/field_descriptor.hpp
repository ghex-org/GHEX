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
#ifndef INCLUDED_GHEX_STRUCTURED_REGULAR_FIELD_DESCRIPTOR_HPP
#define INCLUDED_GHEX_STRUCTURED_REGULAR_FIELD_DESCRIPTOR_HPP

#include "../field_descriptor.hpp"
#include "../field_utils.hpp"
#include "../pack_kernels.hpp"
#include "./domain_descriptor.hpp"
#include <cstring>
#include <cstdint>
#include "../../arch_traits.hpp"

//#define NCTIS 128

namespace gridtools {
namespace ghex {
namespace structured {    
namespace regular {

template<typename T, typename Arch, typename DomainDescriptor, int... Order>
class field_descriptor
: public gridtools::ghex::structured::field_descriptor<T,Arch,DomainDescriptor,Order...>
{
public: // member types
    using base = gridtools::ghex::structured::field_descriptor<T,Arch,DomainDescriptor,Order...>;
    
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
    using serialization_type       = gridtools::ghex::structured::serialization<arch_type,layout_map>;
    using pack_iteration_space     = typename base::pack_iteration_space;
    using unpack_iteration_space   = typename base::unpack_iteration_space;

public: // ctors
    template<typename ExtentArray, typename OffsetArray>
    field_descriptor(
            const domain_descriptor_type& dom_,
            value_type* data_,
            const ExtentArray& offsets_,
            const OffsetArray& extents_,
            unsigned int num_components_ = 1u,
            bool is_vector_field_ = false,
            device_id_type d_id_ = 0)
    : base(dom_, dom_.first(), data_, offsets_, extents_, num_components_, is_vector_field_, d_id_)
    {}
    template<typename ExtentArray, typename OffsetArray, typename Strides>
    field_descriptor(
            const domain_descriptor_type& dom_,
            value_type* data_,
            const ExtentArray& offsets_,
            const OffsetArray& extents_,
            const Strides& strides_,
            unsigned int num_components_,
            bool is_vector_field_,
            device_id_type d_id_)
    : base(dom_, dom_.first(), data_, offsets_, extents_, strides_, num_components_, is_vector_field_, d_id_)
    {}
    field_descriptor(field_descriptor&&) noexcept = default;
    field_descriptor(const field_descriptor&) noexcept = default;
    field_descriptor& operator=(field_descriptor&&) noexcept = default;
    field_descriptor& operator=(const field_descriptor&) noexcept = default;

    template<typename IndexContainer>
    void pack(T* buffer, const IndexContainer& c, void* arg) {
        // loop over pattern's iteration spaces
        for (const auto& is : c) {
            // number of values to pack
            const size_type size = is.size()*base::num_components();
            serialization_type::pack_batch( make_pack_is(is,buffer,size), arg );
            buffer += size;
        }
    }
    
    template<typename IndexContainer>
    void unpack(const T* buffer, const IndexContainer& c, void* arg) {
        // loop over pattern's iteration spaces
        for (const auto& is : c) {
            // number of values to pack
            const size_type size = is.size()*base::num_components();
            serialization_type::unpack_batch( make_unpack_is(is,buffer,size), arg );
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
    unpack_iteration_space make_unpack_is(const IterationSpace& is, const T* buffer, size_type size) {
        return {make_buffer_desc<typename base::template buffer_descriptor<const T*>>(is,buffer,size),
                make_is<typename base::template basic_iteration_space<T*>>(is)};
    }

    template<typename BufferDesc, typename IterationSpace, typename Buffer>
    BufferDesc make_buffer_desc(const IterationSpace& is, Buffer buffer, size_type size) {
        // description of the halo in the buffer
        coordinate_type buffer_offset;
        std::copy(is.global().first().begin(), is.global().first().end(), buffer_offset.begin()); 
        if (has_components::value)
            buffer_offset[dimension::value-1] = 0;
        coordinate_type buffer_extents;
        std::copy(is.global().last().begin(), is.global().last().end(), buffer_extents.begin()); 
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
        std::copy(is.local().first().begin(), is.local().first().end(), data_first.begin());
        std::copy(is.local().last().begin(), is.local().last().end(), data_last.begin());
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

} // namespace regular
} // namespace structured

    /** @brief wrap a N-dimensional array (field) of contiguous memory 
     * @tparam Arch device type the data lives on
     * @tparam Order permutation of the set {0,...,N-1} indicating storage layout (N-1 -> stride=1)
     * @tparam DomainDescriptor domain type
     * @tparam T field value type
     * @tparam Array coordinate-like type
     * @param dom local domain
     * @param data pointer to data
     * @param offsets coordinate of first physical coordinate (not buffer) from the orign of the wrapped N-dimensional array
     * @param extents extent of the wrapped N-dimensional array (including buffer regions)
     * @return wrapped field*/
    template<typename Arch, int... Order, typename DomainDescriptor, typename T, typename Array>
    structured::regular::field_descriptor<T,Arch,DomainDescriptor, Order...>
    wrap_field(const DomainDescriptor& dom, T* data, const Array& offsets, const Array& extents, typename arch_traits<Arch>::device_id_type device_id = 0)
    {
        return {dom, data, offsets, extents, 1, false, device_id};     
    }
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_REGULAR_FIELD_DESCRIPTOR_HPP */

