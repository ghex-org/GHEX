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
#include <memory>
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

		typedef enum {
		    REQ_NONE = 0,
		    REQ_SEND,
		    REQ_RECV
		} t_request_type;

		/** request structure for futures-based comm */
		struct ghex_ucx_request_ft {

		    /* Worker, to which this request was submitted.
		       Needed when we want to cancel it.
		    */
		    ucp_worker_h m_ucp_worker;
		    ucp_worker_h m_ucp_worker_send;
		    t_request_type m_type;
		};

		/** request structure for callback-based comm */
		template<typename Allocator>
		struct ghex_ucx_request_cb {

		    using message_type = shared_message_buffer<Allocator>;

		    ucp_worker_h m_ucp_worker;
		    uint32_t m_peer_rank;
		    uint32_t m_tag;
		    std::function<void(message_type, int, int)> m_cb;
		    message_type m_msg;
		    std::shared_ptr<bool> m_completed;
		    t_request_type m_type;

		    ghex_ucx_request_cb() : 
			m_ucp_worker{nullptr}, 
			m_peer_rank{0}, 
			m_tag{0}, 
			m_msg(0), 
			m_completed(nullptr),
			m_type(REQ_NONE)
		    {}

		    ~ghex_ucx_request_cb(){}

		    ghex_ucx_request_cb(const ghex_ucx_request_cb&) = delete;
		    ghex_ucx_request_cb(ghex_ucx_request_cb &&other) :
			m_ucp_worker{other.m_ucp_worker},
			m_peer_rank{other.m_peer_rank},
			m_tag{other.m_tag},
			m_cb{std::move(other.m_cb)},
			m_msg{std::move(other.m_msg)},
			m_completed{std::move(other.m_completed)},
			m_type{other.m_type}
		    {}

		    ghex_ucx_request_cb& operator=(const ghex_ucx_request_cb &other) = delete;
		};

		/** size of the ghex_ucx_request_cb struct - currently 88 bytes */
#define GHEX_REQUEST_SIZE 88

                /** @brief thin wrapper around UCX Request */
                struct request
                {

		    using req_type = ghex_ucx_request_ft*;
                    req_type m_req = NULL;

		    request(req_type m_req_ = nullptr):
			m_req{m_req_} {}
		    request(const request &&other) = delete;
		    request(request &&other) :
			m_req{other.m_req}
		    {
			other.m_req = nullptr;
		    }

		    ~request()
		    {
			/* user's responsibility to make sure 
			   that this is not called before the comm is completed
			 */
			if(!m_req) return;

			/* TODO: not a good idea to lock the worker
			   in a destructor? */
			CRITICAL_BEGIN(ucp_lock) {
			    
			    /* this shouldn't happen in good code.. */
			    fprintf(stderr, "WARNING: free incomplete request %p\n", (void*)m_req);
			    ucp_request_free(m_req);
			} CRITICAL_END(ucp_lock);
		    }

		    request& operator=(const request &other) = delete;
		    request& operator=(request &&other)
		    {

			if(m_req){
			    CRITICAL_BEGIN(ucp_lock) {

				/* this shouldn't happen in good code.. */
				fprintf(stderr, "WARNING: free incomplete request %p\n", (void*)m_req);
				ucp_request_free(m_req);
			    } CRITICAL_END(ucp_lock);
			}
			
			m_req = other.m_req;
			other.m_req = nullptr;
			return *this;
		    }

                    void wait()
                    {
			if(NULL == m_req) return;
			while (!test());
                    }

                    bool test()
                    {
			ucs_status_t status;
			bool retval = false;

			if(nullptr == m_req) return true;

			ucp_worker_progress(m_req->m_ucp_worker_send);
			ucp_worker_progress(m_req->m_ucp_worker_send);
			ucp_worker_progress(m_req->m_ucp_worker_send);
			CRITICAL_BEGIN(ucp_lock) {

			    /* always progress UCX */
			    ucp_worker_progress(m_req->m_ucp_worker);
			    ucp_worker_progress(m_req->m_ucp_worker);

			    /* ucp_request_check_status has to be locked also:
			       it does access the worker!
			    */

			    /* TODO : are we allowed to call this?
			       m_req might be a send request submitted on 
			       another thread, and hence might access
			       the other threads's send worker... 
			    */
			    
			    /* check request status */
			    status = ucp_request_check_status(m_req);
			    if(UCS_INPROGRESS != status) {
				ucp_request_free(m_req);
				m_req = NULL;
				retval = true;
			    }
			} CRITICAL_END(ucp_lock);

			return retval;
                    }

		    bool cancel(){

			if(nullptr == m_req) return true;

			/* TODO: at this time, send requests cannot be canceled
			   in UCX (1.7.0rc1)
			   https://github.com/openucx/ucx/issues/1162
			*/
			
			/* TODO: the below is only correct for recv requests,
			   or for send requests under the assumption that 
			   the requests cannot be moved between threads.
			   
			   For the send worker we do not use locks, hence
			   if request is canceled on another thread, it might
			   clash with another send being submitted by the owner
			   of ucp_worker
			*/
			if(REQ_SEND == m_req->m_type){
			    return false;
			} else {
			    CRITICAL_BEGIN(ucp_lock){
				ucp_request_cancel(m_req->m_ucp_worker, m_req);
			    } CRITICAL_END(ucp_lock);

			    /* wait for the request to either complete, or be canceled */
			    wait();
			}
			
			return true;
		    }
                };


                /** @brief thin wrapper around UCX Request */
		template<typename Allocator>
                struct request_cb
                {

		    using req_type = ghex_ucx_request_cb<Allocator>*;

                    req_type m_req = nullptr;
		    std::shared_ptr<bool> m_completed;

		    request_cb(req_type req = nullptr): 
			m_req(req)
		    {
			if(m_req) m_completed = m_req->m_completed;
		    }
		    request_cb(const request_cb &other) = delete;
		    request_cb(request_cb &&other) :
			m_req{std::move(other.m_req)},
			m_completed{std::move(other.m_completed)}
		    {
			other.m_req = nullptr;
		    }

		    request_cb& operator=(const request_cb &other) = delete;
		    request_cb& operator=(request_cb &&other)
		    {
			m_req = std::move(other.m_req);
			m_completed = std::move(other.m_completed);
			other.m_req = nullptr;
			return *this;
		    }

                    bool test()
                    {
			if(nullptr == m_req) return true;
			if(*m_completed) {
			    m_req = nullptr;
			    m_completed = nullptr;
			    return true;
			}
			return false;
                    }

		    bool cancel(){

			if(nullptr == m_req) return true;

			/* TODO: at this time, send requests cannot be canceled
			   in UCX (1.7.0rc1)
			   https://github.com/openucx/ucx/issues/1162
			*/
			if(REQ_SEND == m_req->m_type) return false;

			/* TODO: the below is only correct for recv requests,
			   or for send requests under the assumption that 
			   the requests cannot be moved between threads.
			   
			   For the send worker we do not use locks, hence
			   if request is canceled on another thread, it might
			   clash with another send being submitted by the owner
			   of ucp_worker
			*/
			CRITICAL_BEGIN(ucp_lock){
			    if(!(*m_completed)) 
				ucp_request_cancel(m_req->m_ucp_worker, m_req);
			    m_req = nullptr;
			    m_completed = nullptr;
			} CRITICAL_END(ucp_lock);
			return true;
		    }
                };
            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_REQUEST_HPP */
