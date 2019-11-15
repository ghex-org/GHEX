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
#ifndef INCLUDED_GHEX_HALO_HPP
#define INCLUDED_GHEX_HALO_HPP

#include "./arch_traits.hpp"

namespace gridtools {
    namespace ghex {

        // forward declaration
        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern;

        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern_container;

        /** @brief class which combines pattern, field and domain into a halo handle
          * @tparam Pattern pattern type
          * @tparam Field field type*/
        template<typename Pattern, typename Field>
        struct halo_t
        {
        public: // member types
            using pattern_type             = Pattern;
            using transport_type           = typename pattern_type::transport_type; 
            using grid_type                = typename pattern_type::grid_type; 
            using domain_id_type           = typename pattern_type::domain_id_type;
            using pattern_container_type   = typename pattern_type::pattern_container_type;
            using field_type               = Field;
            using arch_type                = typename field_type::arch_type;
            using value_type               = typename field_type::value_type; 
            using device_id_type           = typename arch_traits<arch_type>::device_id_type;

        private: // friend class
            friend class pattern_container<transport_type,grid_type,domain_id_type>;

        private: // members
            const pattern_type* m_p;
            domain_id_type      m_domain_id;
            field_type*         m_field;
            device_id_type      m_device_id;

        private: // private ctor
            halo_t(const pattern_type& p, domain_id_type domain_id_, field_type& field) noexcept
            : m_p{&p}
            , m_domain_id{domain_id_}
            , m_field{&field}
            {}

        public: // copy and move ctors
            halo_t(const halo_t&) noexcept = default;
            halo_t(halo_t&&) noexcept = default;

        public: // member functions
            const pattern_type& pattern() const noexcept { return *m_p; }
            const pattern_container_type& pattern_container() const noexcept { return m_p->container(); }
            
            field_type& field() noexcept { return *m_field; }
            const field_type& field() const noexcept { return *m_field; }
            
            domain_id_type domain_id() const noexcept { return m_domain_id; }
            device_id_type device_id() const noexcept { return m_field->device_id(); }
        };

    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_HALO_HPP */

