/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_BUFFER_INFO_HPP
#define INCLUDED_BUFFER_INFO_HPP

#include <vector>

namespace gridtools {

    // forward declaration
    template<typename P, typename GridType, typename DomainIdType>
    class pattern;
    template<typename P, typename GridType, typename DomainIdType>
    class pattern_container;
    
    // forward declaration
    template<typename Pattern, typename Device, typename Field>
    struct buffer_info;

    /** @brief ties together field, pattern and device
     * @tparam P message transport protocol
     * @tparam GridType grid tag type
     * @tparam DomainIdType domain id type
     * @tparam Device device type
     * @tparam Field field descriptor type */
    template<typename P, typename GridType, typename DomainIdType, typename Device, typename Field>
    struct buffer_info<pattern<P,GridType,DomainIdType>, Device, Field>
    {
    public: // member types
        using pattern_type             = pattern<P,GridType,DomainIdType>;
        using pattern_container_type   = pattern_container<P,GridType,DomainIdType>;
        using device_type              = Device;
        using field_type               = Field;
        using device_id_type           = typename device_type::device_id_type;
        using value_type               = typename field_type::value_type; 
   
    private: // friend class
        friend class pattern<P,GridType,DomainIdType>;

    private: // private ctor
        buffer_info(const pattern_type& p, field_type& field, device_id_type id) noexcept
        :   m_p{&p}, m_field{&field}, m_id{id} { }

    public: // copy and move ctors
        buffer_info(const buffer_info&) noexcept = default;
        buffer_info(buffer_info&&) noexcept = default;

    public: // member functions
        device_id_type device_id() const noexcept { return m_id; }
        const pattern_type& get_pattern() const noexcept { return *m_p; }
        const pattern_container_type& get_pattern_container() const noexcept { return m_p->container(); }
        field_type& get_field() noexcept { return *m_field; }

    private: // members
        const pattern_type* m_p;
        field_type* m_field;
        device_id_type m_id;
    };

} // namespace gridtools


#endif /* INCLUDED_BUFFER_INFO_HPP */

