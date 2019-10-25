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
#ifndef INCLUDED_GHEX_TL_UCX_ADDRESS_HPP
#define INCLUDED_GHEX_TL_UCX_ADDRESS_HPP

#include <ucp/api/ucp.h>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

                /** @brief thin wrapper around UCX Request */
                struct address
                {
		    ucp_worker_h  worker;
		    ucp_address_t *addr;
		    int size;

		    address(ucp_worker_h _worker, ucp_address_t *_addr, int _size):
			worker{_worker}, addr{_addr}, size{size}
		    {}

		    ~address(){
			ucp_worker_release_address(worker, addr);
		    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_ADDRESS_HPP */
