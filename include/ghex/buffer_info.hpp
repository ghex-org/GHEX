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
#ifndef INCLUDED_GHEX_BUFFER_INFO_HPP
#define INCLUDED_GHEX_BUFFER_INFO_HPP

#include <vector>
#include "./arch_traits.hpp"

namespace gridtools {

    namespace ghex {

        // forward declaration
        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern;
        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern_container;
        
        // forward declaration
        template<typename Pattern, typename Arch, typename Field>
        struct buffer_info;

        /** @brief ties together field, pattern and device
         * @tparam Transport message transport protocol
         * @tparam GridType grid tag type
         * @tparam DomainIdType domain id type
         * @tparam Arch device type
         * @tparam Field field descriptor type */
        template<typename Transport, typename GridType, typename DomainIdType, typename Arch, typename Field>
        struct buffer_info<pattern<Transport,GridType,DomainIdType>, Arch, Field>
        {
        public: // member types
            using pattern_type             = pattern<Transport,GridType,DomainIdType>;
            using pattern_container_type   = pattern_container<Transport,GridType,DomainIdType>;
            using arch_type              = Arch;
            using field_type               = Field;
            using device_id_type           = typename arch_traits<arch_type>::device_id_type;
            using value_type               = typename field_type::value_type; 
       
        private: // friend class
            friend class pattern<Transport,GridType,DomainIdType>;

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

    } // namespace ghex

} // namespace gridtools


#endif /* INCLUDED_GHEX_BUFFER_INFO_HPP */

