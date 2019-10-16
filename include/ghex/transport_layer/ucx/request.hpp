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
#ifndef INCLUDED_GHEX_TL_UCX_REQUEST_HPP
#define INCLUDED_GHEX_TL_UCX_REQUEST_HPP

#include <functional>
#include <ucp/api/ucp.h>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

		/** request structure for futures-based comm */
		struct ghex_ucx_request {
		    ucp_worker_h ucp_worker; // worker thread handling this request
		};
		
		/** request structure for callback-based comm */
		template<typename MsgType>
		struct ghex_ucx_request_cb {
		    uint32_t peer_rank;
		    uint32_t tag; 
		    std::function<void(int, int, const MsgType&)> cb;
		    MsgType h_msg;
		};

		/** size of the above struct for actual MsgType */
		#define GHEX_REQUEST_SIZE 48

                /** @brief thin wrapper around UCX Request */
                struct request
                {

		    using req_type = ghex_ucx_request*;
                    req_type m_req = NULL;

                    void wait()
                    {
			if(NULL == m_req) return;
			while (!test());
                    }

                    bool test()
                    {
			ucs_status_t status;
			if(NULL == m_req) return true;
			status = ucp_request_check_status(m_req);
			if(status != UCS_INPROGRESS) return true;

			/* progress UCX */
			CRITICAL_BEGIN(ucp) {
			    ucp_worker_progress(m_req->ucp_worker);
			} CRITICAL_END(ucp);

			status = ucp_request_check_status(m_req);
			if(status != UCS_INPROGRESS) return true;
	
			return false;
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */
