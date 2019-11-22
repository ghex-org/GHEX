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

#ifdef USE_RAW_SHARED_MESSAGE
#include "../raw_shared_message_buffer.hpp"
#else
#include "../shared_message_buffer.hpp"
#endif

#include "ucp_lock.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

		/** request structure for futures-based comm */
		struct ghex_ucx_request_ft {
		    ucp_worker_h m_ucp_worker;
		    ucp_worker_h m_ucp_worker_send;
		};

		/** request structure for callback-based comm */
		template<typename Allocator>
		struct ghex_ucx_request_cb {

		    using message_type = shared_message_buffer<Allocator>;

		    uint32_t m_peer_rank;
		    uint32_t m_tag;
		    std::function<void(message_type, int, int)> m_cb;
		    message_type m_msg;
		    unsigned char m_initialized; // to handle early completion

		    ghex_ucx_request_cb() : m_peer_rank{0}, m_tag{0}, m_msg(0), m_initialized(0) {}
		    ~ghex_ucx_request_cb(){}

		    ghex_ucx_request_cb(const ghex_ucx_request_cb&) = delete;
		    ghex_ucx_request_cb(ghex_ucx_request_cb &&other) :
			m_peer_rank{other.m_peer_rank},
			m_tag{other.m_tag},
			m_cb{std::move(other.m_cb)},
			m_msg{std::move(other.m_msg)},
			m_initialized(std::move(other.m_initialized))
		    {}

		    ghex_ucx_request_cb& operator=(const ghex_ucx_request_cb &other) = delete;
		};

		/** size of the above struct for actual MsgType */
#define GHEX_REQUEST_SIZE 64

                /** @brief thin wrapper around UCX Request */
                struct request
                {

		    using req_type = ghex_ucx_request_ft*;
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

			ucp_worker_progress(m_req->m_ucp_worker_send);
			CRITICAL_BEGIN(ucp_lock) {

			    /* always progress UCX */
			    ucp_worker_progress(m_req->m_ucp_worker);

			    /* ucp_request_check_status has to be locked also:
			       it does access the worker!
			    */
			    status = ucp_request_check_status(m_req);
			    if(status != UCS_INPROGRESS) {
				ucp_request_free(m_req);
				m_req = NULL;
				retval = true;
			    }
			} CRITICAL_END(ucp_lock);

			return retval;
                    }

		    bool cancel(){

			/* TODO: need to know which worker to use to cancel? */
			CRITICAL_BEGIN(ucp_lock){
			    ucp_request_cancel(m_req->m_ucp_worker, m_req);
			} CRITICAL_END(ucp_lock);
			return true;
		    }
                };
            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */
