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
#ifndef INCLUDED_GHEX_COMMUNICATION_OBJECT_IPR_HPP
#define INCLUDED_GHEX_COMMUNICATION_OBJECT_IPR_HPP

namespace gridtools {

    namespace ghex {

        // Forward declarations:

        /** @brief communication object responsible for exchanging halo data with in-place receive.
         * Allocates storage depending on the device type and device id of involved fields.
         * @tparam Communicator communicator type
         * @tparam GridType grid tag type
         * @tparam DomainIdType domain id type*/
        template<typename Communicator, typename GridType, typename DomainIdType>
        class communication_object_ipr;

        /** @brief handle type for waiting on asynchronous communication processes.
         * The wait function is stored in a member.
         * @tparam Communicator communicator type
         * @tparam GridType grid tag type
         * @tparam DomainIdType domain id type*/
        template<typename Communicator, typename GridType, typename DomainIdType>
        class communication_handle_ipr;

        namespace detail {
            // Internal forward declaration
            template<typename Communicator, typename GridType, typename DomainIdType>
            struct make_communication_object_ipr_impl;
        } // namespace detail

        /** @brief creates a communication object based on the pattern type
         * @tparam PatternContainer pattern type
         * @param comm transport communicator
         * @return communication object */
        template<typename PatternContainer>
        auto make_communication_object_ipr(typename PatternContainer::value_type::communicator_type comm) {
            using communicator_type = typename PatternContainer::value_type::communicator_type;
            using grid_type = typename PatternContainer::value_type::grid_type;
            using domain_id_type = typename PatternContainer::value_type::domain_id_type;
            return detail::make_communication_object_ipr_impl<communicator_type, grid_type, domain_id_type>::apply(comm);
        }

    } // namespace ghex
        
} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMUNICATION_OBJECT_IPR_HPP */
