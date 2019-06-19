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

    template<typename P, typename GridType, typename DomainIdType>
    class pattern;
    
    template<typename Pattern, typename Device, typename Field>
    struct buffer_info {};

    template<typename P, typename GridType, typename DomainIdType, typename Device, typename Field>
    struct buffer_info<pattern<P,GridType,DomainIdType>, Device, Field>
    {
    public:
        using pattern_type             = pattern<P,GridType,DomainIdType>;
        using device_type              = Device;
        using field_type               = Field;
        using device_id_type           = typename device_type::id_type;
        using value_type               = typename field_type::value_type; 

    public:
        buffer_info(const pattern_type& p, field_type& field, device_id_type id)
        :   m_p{p}, m_field{field}, m_id{id}
        { }

        buffer_info(const buffer_info&) noexcept = default;
        buffer_info(buffer_info&&) noexcept = default;

        device_id_type device_id() const noexcept { return m_id; }
        const pattern_type& get_pattern() const noexcept { return m_p; }
        field_type& get_field() noexcept { return m_field; }

    private:
        const pattern_type& m_p;
        field_type& m_field;
        device_id_type m_id;
    };

} // namespace gridtools


#endif /* INCLUDED_BUFFER_INFO_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

