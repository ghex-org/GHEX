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

#include "pattern.hpp"
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
        //using extended_domain_id_type  = typename pattern_type::extended_domain_id_type;
        using device_id_type           = typename device_type::id_type;
        using value_type               = typename field_type::value_type; 


        //using index_container_type    = typename pattern_type::index_container_type;
        //using pack_function_type      = std::function<void(void*,const index_container_type&)>;
        //using unpack_function_type    = std::function<void(void*,const index_container_type&)>;
        //using memory_type             = std::vector<typename device_type::template vector_type<char> *>;

    public:
        buffer_info(const pattern_type& p, field_type& field, device_id_type id)
        :   m_p{p}, m_field{field}, m_id{id}
        {
            /*m_sizes_recv.reserve(p.recv_halos().size());
            for (const auto& c : p.recv_halos())
                m_sizes_recv.push_back(
                    static_cast<std::size_t>(pattern_type::num_elements(c.second))*
                    static_cast<std::size_t>(sizeof(value_type)));
            m_sizes_send.reserve(p.send_halos().size());
            for (const auto& c : p.send_halos())
                m_sizes_send.push_back(
                    static_cast<std::size_t>(pattern_type::num_elements(c.second))*
                    static_cast<std::size_t>(sizeof(value_type)));*/
        }

        buffer_info(const buffer_info&) noexcept = default;
        buffer_info(buffer_info&&) noexcept = default;

        device_id_type device_id() const noexcept { return m_id; }
        //const std::vector<std::size_t>& sizes_recv() const noexcept { return m_sizes_recv; }
        //const std::vector<std::size_t>& sizes_send() const noexcept { return m_sizes_send; }
        const pattern_type& get_pattern() const noexcept { return m_p; }
        field_type& get_field() noexcept { return m_field; }

    private:
        const pattern_type& m_p;
        field_type& m_field;
        device_id_type m_id;
        //std::vector<std::size_t> m_sizes_recv;
        //std::vector<std::size_t> m_sizes_send;

        //pack_function_type m_pack;
        //unpack_function_type m_unpack;
        //memory_type m_memory;
    };

} // namespace gridtools


#endif /* INCLUDED_BUFFER_INFO_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

