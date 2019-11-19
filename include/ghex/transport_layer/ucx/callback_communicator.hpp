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
#ifndef INCLUDED_GHEX_TL_UCX_CALLBACK_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_UCX_CALLBACK_COMMUNICATOR_HPP

#include <functional>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <tuple>
#include <boost/callable_traits.hpp>
#include <boost/optional.hpp>

#include <ucp/api/ucp.h>

#include "communicator.hpp"
#include "../callback_communicator.hpp"
#include "../../common/debug.hpp"

#ifdef USE_RAW_SHARED_MESSAGE
#include "../raw_shared_message_buffer.hpp"
#else
#include "../shared_message_buffer.hpp"
#endif

#include "request.hpp"
#include "future.hpp"

namespace gridtools
{
    namespace ghex
    {
        namespace tl {

	    /** completion callbacks registered in UCX, defined later */
	    template <typename Allocator>
	    void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
	    template <typename Allocator>
	    void ghex_tag_send_callback(void *request, ucs_status_t status);


            /** This class specializes the callbac_communicator for the UCX transport communicator
	     *
	     * callback_communicator is a class to dispatch send and receive operations to. Each operation can 
	     * optionally be tied to a user defined callback function / function object. The payload of each 
	     * send/receive operation must be a ghex::shared_message_buffer<Allocator>. 
	     * This class will keep a (shallow) copy of each message, thus it is safe to release the message at 
	     * the caller's site.
	     *
	     * The user defined callback must define void operator()(message_type,rank_type,tag_type), where
	     * message_type is a shared_message_buffer that can be cheaply copied/moved from within the callback body 
	     * if needed.
	     *
	     * The communication must be explicitely progressed using the member function progress.
	     *
	     * An instance of this class is 
	     * - a move-only.
	     * - thread-safe
	     *
	     * @tparam Allocator    allocator type used for allocating shared message buffers */

            template<class Allocator>
	    class callback_communicator<communicator<ucx_tag>, Allocator>: public communicator<ucx_tag>
            {
            public: // member types
                
                using communicator_type = communicator<ucx_tag>;
                using allocator_type    = Allocator;
                using message_type      = shared_message_buffer<allocator_type>;

            private: // members

		/* Request structure to transfer needed request info to the
		 * recv callback when early recv is called from inside ucp_tag_recv_nb
		 */
		int m_early_completion = 0;
		ucx::ghex_ucx_request_cb<Allocator> m_early_req;

		/* A list of completed request. Instead of calling the user callbacks immediately
		 * from inside the UCX completion callbacks, we add the completed requests to this list.
		 * The user callbacks are then called explicitly from the progress() function.
		 * The point is to avoid calling the user callback from inside a locked region
		 */
#ifndef USE_HEAVY_CALLBACKS
		std::vector<ucx::ghex_ucx_request_cb<Allocator>> m_completed;
#endif
		
            public: // ctors

                callback_communicator() {}
                callback_communicator(const callback_communicator&) = delete;
                callback_communicator(callback_communicator&&) = default;
                ~callback_communicator() = default;

            public: // send

                /** @brief Send a message to a destination with the given tag and register a callback which will be 
		 * invoked when the send operation is completed.
		 * @tparam CallBack User defined callback class which defines 
		 *                  void Callback::operator()(message_type,rank_type,tag_type)
		 * @param msg Message to be sent
		 * @param dst Destination of the message
		 * @param tag Tag associated with the message
		 * @param cb  Callback function object */
		template <typename CallBack>
		void send(const message_type &msg, rank_type dst, tag_type tag, CallBack &&cb)
		{
		    ucp_ep_h ep;
		    ucs_status_ptr_t status;
		    uintptr_t istatus;
		    ucx::ghex_ucx_request_cb<Allocator> *ghex_request;

		    ep = rank_to_ep(dst);
		    
		    /* send with callback */
		    status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
					     GHEX_MAKE_SEND_TAG(tag, m_rank), ghex_tag_send_callback<Allocator>);

		    // TODO !! C++ doesn't like it..
		    istatus = (uintptr_t)status;
		    if(UCS_OK == (ucs_status_t)(istatus)){

			/* early completed */
#ifdef USE_HEAVY_CALLBACKS
			cb(msg, dst, tag);
#else
			early = true;
#endif
		    } else if(!UCS_PTR_IS_ERR(status)) {

			/* prepare the request */
			m_early_req.m_peer_rank = dst;
			m_early_req.m_tag = tag;
			m_early_req.m_cb = std::forward<CallBack>(cb);
			m_early_req.m_msg = msg;

			/* fill in request data */
			ghex_request = (ucx::ghex_ucx_request_cb<Allocator>*)status;			    
			new(ghex_request) ucx::ghex_ucx_request_cb<Allocator>(std::move(m_early_req));
		    } else {
			ERR("ucp_tag_send_nb failed");
		    }

#ifndef USE_HEAVY_CALLBACKS
		    /* call the callback outside of the critical region */
		    if(early){
		    	cb(msg, dst, tag);
		    }
#endif
		}


            public: // recieve

                /** @brief Receive a message from a source rank with the given tag and register a callback which will
		 * be invoked when the receive operation is completed.
		 * @tparam CallBack User defined callback class which defines 
		 *                  void Callback::operator()(message_type,rank_type,tag_type)
		 * @param msg Message where data will be received
		 * @param src Source of the message
		 * @param tag Tag associated with the message
		 * @param cb  Callback function object */
		template <typename CallBack>
		void recv(message_type &msg, rank_type src, tag_type tag, CallBack &&cb)
		{
		    ucp_tag_t ucp_tag, ucp_tag_mask;
		    ucs_status_ptr_t status;
		    ucx::ghex_ucx_request_cb<Allocator> *ghex_request;

		    CRITICAL_BEGIN(ucp_lock) {

			/* set request init data - it might be that the recv completes inside ucp_tag_recv_nb */
			/* and the callback is called earlier than we initialize the data inside it */
			m_early_req.m_peer_rank = src;
			m_early_req.m_tag = tag;
			m_early_req.m_cb = std::function<void(message_type, int, int)>(cb);
			m_early_req.m_msg = msg;

			m_early_completion = 1;
			GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
			status = ucp_tag_recv_nb(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 ucp_tag, ucp_tag_mask, ghex_tag_recv_callback<Allocator>);
			m_early_completion = 0;

			if(!UCS_PTR_IS_ERR(status)) {

			    ucs_status_t rstatus;
			    rstatus = ucp_request_check_status (status);
			    if(rstatus != UCS_INPROGRESS){

				/* early completed */
				ucp_request_free(status);
			    } else {

				/* fill in request data */
				ghex_request = (ucx::ghex_ucx_request_cb<Allocator> *)status;
				new(ghex_request) ucx::ghex_ucx_request_cb<Allocator>(std::move(m_early_req));
			    }
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp_lock);
		}

                /** @brief Progress the communication. This function progresses the UCX backend. 
		    For completed requests, the associated callbacks are called.
		 * @return returns the number of progressed requests. */
		unsigned progress()
		{
		    unsigned p = communicator<ucx_tag>::progress();
		    
#ifndef USE_HEAVY_CALLBACKS
		    /* call the callbacks of completed requests outside of the critical region */
		    while(m_completed.size()){
		    	ucx::ghex_ucx_request_cb<Allocator> req = std::move(m_completed.back());
			m_completed.pop_back();
		    	req.m_cb(std::move(req.m_msg), req.m_peer_rank, req.m_tag);
		    }
#endif
		    return p;
		}

		/** completion callbacks registered in UCX
		 *  require access to private properties.
		 */
		friend void ghex_tag_recv_callback<Allocator>(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
		friend void ghex_tag_send_callback<Allocator>(void *request, ucs_status_t status);
	    };


	    /** request completion callbacks registered in UCX 
	     */    
	    template <typename Allocator>
	    void ghex_tag_recv_callback(void *request, ucs_status_t, ucp_tag_recv_info_t *info)
	    {
		uint32_t peer_rank = GHEX_GET_SOURCE(info->sender_tag); // should be the same as r->peer_rank
		uint32_t tag = GHEX_GET_TAG(info->sender_tag);          // should be the same as r->tagx

		callback_communicator<communicator<ucx_tag>, Allocator> *pc = 
		    pc = reinterpret_cast<callback_communicator<communicator<ucx_tag>, Allocator> *>(ucx::pcomm);

		/* pointer to request data */
		ucx::ghex_ucx_request_cb<Allocator> *preq;

		if(pc->m_early_completion){
#ifdef USE_HEAVY_CALLBACKS
		    preq = &pc->m_early_req;
		    preq->m_cb(std::move(preq->m_msg), peer_rank, tag);
#else
		    pc->m_completed.push_back(std::move(pc->m_early_req));
#endif
		    /* do not free the request - it has to be freed after tag_send_nb */
		} else {

		    preq = reinterpret_cast<ucx::ghex_ucx_request_cb<Allocator>*>(request);
#ifdef USE_HEAVY_CALLBACKS
		    preq->m_cb(std::move(preq->m_msg), peer_rank, tag);
#else
		    pc->m_completed.push_back(std::move(*preq));
#endif
		    ucp_request_free(request);
		}
	    }

	    template <typename Allocator>
	    void ghex_tag_send_callback(void *request, ucs_status_t)
	    {
		callback_communicator<communicator<ucx_tag>, Allocator> *pc = 
		    pc = reinterpret_cast<callback_communicator<communicator<ucx_tag>, Allocator> *>(ucx::pcomm);

		ucx::ghex_ucx_request_cb<Allocator> *preq =
		    reinterpret_cast<ucx::ghex_ucx_request_cb<Allocator>*>(request);
		
#ifdef USE_HEAVY_CALLBACKS
		preq->m_cb(std::move(preq->m_msg), preq->m_peer_rank, preq->m_tag);
#else
		pc->m_completed.push_back(std::move(*preq));
#endif		
		ucp_request_free(request);
	    }
        } // namespace tl
    } // namespace ghex
}// namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_CALLBACK_COMMUNICATOR_HPP */

