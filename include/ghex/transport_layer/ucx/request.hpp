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

#include "locks.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

		/** request structure for futures-based comm */
		struct ghex_ucx_request {
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

		/** this is defined in ucx communicator.hpp.
		    Requests have no access to the worker, and we
		    need to progess the engine here 
		*/
		extern void worker_progress();

#ifdef THREAD_MODE_MULTIPLE
#ifndef USE_OPENMP_LOCKS
		extern lock_t ucp_lock;
#endif
#endif

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
			bool retval = false;
			
			if(NULL == m_req) return true;

			/* ucp_request_check_status has to be locked also:
			   it does access the worker!
			*/
			CRITICAL_BEGIN(ucp_lock) {
			    status = ucp_request_check_status(m_req);
			    if(status != UCS_INPROGRESS) {
				ucp_request_free(m_req);
				m_req = NULL;
				retval = true;
			    } else {

				/* progress UCX */
				worker_progress();

				status = ucp_request_check_status(m_req);
				if(status != UCS_INPROGRESS) {
				    ucp_request_free(m_req);
				    m_req = NULL;
				    retval = true;
				}
			    }
			} CRITICAL_END(ucp_lock);
			// sched_yield();
			return retval;
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */
