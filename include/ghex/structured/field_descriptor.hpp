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
#ifndef INCLUDED_GHEX_STRUCTURED_FIELD_DESCRIPTOR_HPP
#define INCLUDED_GHEX_STRUCTURED_FIELD_DESCRIPTOR_HPP

#include <algorithm>
#include "./field_utils.hpp"
#include "../common/utils.hpp"
#include "../arch_traits.hpp"
#include <gridtools/common/array.hpp>
#include <gridtools/common/layout_map.hpp>

namespace gridtools {
namespace ghex {
namespace structured {
    
template<typename T, typename Arch, typename DomainDescriptor, int... Order>
class field_descriptor
{
public: // member types
    using value_type             = T;
    using arch_type              = Arch;
    using device_id_type         = typename arch_traits<arch_type>::device_id_type;
    using domain_descriptor_type = DomainDescriptor;
    using dimension              = std::integral_constant<std::size_t,sizeof...(Order)>;
    using has_components         = std::integral_constant<bool,
        (dimension::value > domain_descriptor_type::dimension::value)>;
    using layout_map             = ::gridtools::layout_map<Order...>;
    using domain_id_type         = typename DomainDescriptor::domain_id_type;
    using scalar_coordinate_type = typename domain_descriptor_type::coordinate_type::value_type;
    using coordinate_type        = ::gridtools::array<scalar_coordinate_type, dimension::value>;
    using size_type              = unsigned int;
    using strides_type           = ::gridtools::array<size_type, dimension::value>;

    // holds buffer information
    template<typename Pointer>
    struct buffer_descriptor {
        Pointer m_ptr;
        const coordinate_type m_first;
        const strides_type m_strides;
        const size_type m_size;
    };
    
    // holds halo iteration space information
    template<typename Pointer>
    struct basic_iteration_space {
        Pointer m_ptr;
        const coordinate_type m_domain_first;
        const coordinate_type m_offset;
        const coordinate_type m_first;
        const coordinate_type m_last;
        const strides_type m_strides;
        const strides_type m_local_strides;
    };

    struct pack_iteration_space {
        using value_t = T;
        using coordinate_t = coordinate_type;
        const buffer_descriptor<T*> m_buffer_desc;
        const basic_iteration_space<const T*> m_data_is;

        /** @brief accesses buffer at specified local coordinate
          * @param coord in local coordinate system
          * @return reference to the value in the buffer */
        GT_FUNCTION
        T& buffer(const coordinate_type& coord) const noexcept {
            // compute buffer coordinates, relative to the buffer origin
            const coordinate_type buffer_coord = coord - m_data_is.m_first;
            // dot product with strides to compute address
            const auto memory_location = dot(m_buffer_desc.m_strides, buffer_coord);
            return *reinterpret_cast<T*>(
                reinterpret_cast<char*>(m_buffer_desc.m_ptr) + memory_location);
        }

        /** @brief accesses field at specified local coordinate
          * @param coord in local coordinate system
          * @return const reference to the value in the field */
        GT_FUNCTION
        const T& data(const coordinate_type& coord) const noexcept {
            // make data memory coordinates from local coordinates
            const coordinate_type data_coord = coord + m_data_is.m_offset;
            // dot product with strides to compute address
            const auto memory_location = dot(m_data_is.m_strides, data_coord);
            return *reinterpret_cast<const T*>(
                reinterpret_cast<const char*>(m_data_is.m_ptr) + memory_location);
        }

    };

    struct unpack_iteration_space {
        using value_t = T;
        using coordinate_t = coordinate_type;
        const buffer_descriptor<const T*> m_buffer_desc;
        const basic_iteration_space<T*> m_data_is;

        /** @brief accesses buffer at specified local coordinate
          * @param coord in local coordinate system
          * @return value in the buffer */
        GT_FUNCTION
        const T& buffer(const coordinate_type& coord) const noexcept {
            // compute buffer coordinates, relative to the buffer origin
            const coordinate_type buffer_coord = coord - m_data_is.m_first;
            // dot product with strides to compute address
            const auto memory_location = dot(m_buffer_desc.m_strides, buffer_coord);
            return *reinterpret_cast<const T*>(
                reinterpret_cast<const char*>(m_buffer_desc.m_ptr) + memory_location);
        }

        /** @brief accesses field at specified local coordinate
          * @param local coordinate system
          * @return reference to the value in the field */
        GT_FUNCTION
        T& data(const coordinate_type& coord) const noexcept {
            // make data memory coordinates from local coordinates
            const coordinate_type data_coord = coord + m_data_is.m_offset;
            // dot product with strides to compute address
            const auto memory_location = dot(m_data_is.m_strides, data_coord);
            return *reinterpret_cast<T*>(
                reinterpret_cast<char*>(m_data_is.m_ptr) + memory_location);
        }
    };

protected: // members
    domain_descriptor_type m_dom;             ///< domain descriptor
    value_type*            m_data;            ///< pointer to data
    coordinate_type        m_dom_first;       ///< global coordinate of first domain cell
    coordinate_type        m_offsets;         ///< offset from beginning of memory to the first domain cell
    coordinate_type        m_extents;         ///< extent of memory (including halos)
    size_type              m_num_components;  ///< number of components 
    bool                   m_is_vector_field; ///< true if this field describes a vector field
    device_id_type         m_device_id;       ///< device id
    strides_type           m_byte_strides;    ///< memory strides in bytes

public: // ctors
    template<typename DomainArray, typename OffsetArray, typename ExtentArray>
    field_descriptor(
        const domain_descriptor_type& dom_,
        const DomainArray& dom_first_,
        value_type* data_,
        const OffsetArray& offsets_,
        const ExtentArray& extents_,
        unsigned int num_components_ = 1u,
        bool is_vector_field_ = false,
        device_id_type d_id_ = 0)
    : m_dom{dom_}
    , m_data{data_}
    , m_num_components{num_components_}
    , m_is_vector_field{is_vector_field_}
    , m_device_id{d_id_}
    {
        // check if components are allowed
        if (m_num_components == 0u)
            throw std::runtime_error("number of components must be greater than 0");
        if (!has_components::value && m_num_components > 1u)
            throw std::runtime_error("this field cannot have more than 1 components");
        // global coordinate of the first physical node
        std::copy(dom_first_.begin(), dom_first_.end(), m_dom_first.begin());
        //if (has_components::value) m_dom_first[dimension::value-1] = 0;
        // offsets from beginning of data to the first physical (local) node
        std::copy(offsets_.begin(), offsets_.end(), m_offsets.begin());
        // extents of the field including buffers
        std::copy(extents_.begin(), extents_.end(), m_extents.begin());
        if (has_components::value) {
            m_dom_first[dimension::value-1] = 0;
            m_offsets[dimension::value-1] = 0;
            m_extents[dimension::value-1] = num_components_;
        }
        // compute strides in bytes
        detail::compute_strides<dimension::value>::template
            apply<layout_map,value_type>(m_extents,m_byte_strides,0u);
    }
    template<typename DomainArray, typename OffsetArray, typename ExtentArray, typename Strides>
    field_descriptor(
        const domain_descriptor_type& dom_,
        const DomainArray& dom_first_,
        value_type* data_,
        const OffsetArray& offsets_,
        const ExtentArray& extents_,
        const Strides& strides_,
        unsigned int num_components_ = 1u,
        bool is_vector_field_ = false,
        device_id_type d_id_ = 0)
    : field_descriptor(dom_, dom_first_, data_, offsets_, extents_, num_components_, is_vector_field_, d_id_)
    {
        for (unsigned int i=0u; i<dimension::value; ++i)
            m_byte_strides[i] = strides_[i];
    }

    field_descriptor(field_descriptor&&) noexcept = default;
    field_descriptor(const field_descriptor&) noexcept = default;
    field_descriptor& operator=(field_descriptor&&) noexcept = default;
    field_descriptor& operator=(const field_descriptor&) noexcept = default;

public: // member functions
    /** @brief returns the device id */
    typename arch_traits<arch_type>::device_id_type device_id() const { return m_device_id; }
    /** @brief returns the domain id */
    domain_id_type domain_id() const noexcept { return m_dom.domain_id(); }
    /** @brief returns the field 4-D extents (c,x,y,z) */
    const coordinate_type& extents() const noexcept { return m_extents; }
    /** @brief returns the field 4-D offsets (c,x,y,z) */
    const coordinate_type& offsets() const noexcept { return m_offsets; }
    /** @brief returns the field 4-D byte strides (c,x,y,z) */
    const strides_type& byte_strides() const noexcept { return m_byte_strides; }
    /** @brief returns the pointer to the data */
    value_type* data() const { return m_data; }
    /** @brief set a new pointer to the data */
    void set_data(value_type* ptr) { m_data = ptr; }
    /** @brief returns the number of components c */
    int num_components() const noexcept { return m_num_components; }
    /** @brief returns true if this field describes a vector field */
    bool is_vector_field() const noexcept { return m_is_vector_field; }
        
    /** @brief access operator
     * @param x coordinate vector with respect to offset specified in constructor
     * @return reference to value */
    GT_FUNCTION
    value_type& operator()(const coordinate_type& x) {
        return *reinterpret_cast<T*>((char*)m_data +dot(x,m_byte_strides));
    }
    GT_FUNCTION
    const value_type& operator()(const coordinate_type& x) const {
        return *reinterpret_cast<const T*>((const char*)m_data +dot(x,m_byte_strides));
    }

    /** @brief access operator
     * @param is coordinates with respect to offset specified in constructor
     * @return reference to value */
    template<typename... Is>
    GT_FUNCTION
    value_type& operator()(Is&&... is) {
        return *reinterpret_cast<T*>((char*)m_data+dot(coordinate_type{is...}+m_offsets,m_byte_strides));
    }
    template<typename... Is>
    GT_FUNCTION
    const value_type& operator()(Is&&... is) const {
        return *reinterpret_cast<const T*>((const char*)m_data+dot(coordinate_type{is...}+
            m_offsets,m_byte_strides));
    }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_FIELD_DESCRIPTOR_HPP */
