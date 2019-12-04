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
#ifndef INCLUDED_GHEX_TL_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_COMMUNICATOR_HPP

namespace gridtools {

    namespace ghex {
    
        namespace tl {

            /** @brief communicator class which exposes basic communication primitives 
              * @tparam TransportTag transport protocol tag */
            template<typename TransportTag,typename ThreadPrimitives>
            class communicator; 
            
                // concept
                // -------
                // using transport_type = P;
                // using handle_type   = ...;
                // using address_type  = ...;
                // template<typename T>
                // using future = ...;

            /** @brief mpi transport tag */
            struct mpi_tag {};


        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_COMMUNICATOR_HPP */

