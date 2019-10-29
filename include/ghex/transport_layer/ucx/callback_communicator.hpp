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

#include "locks.hpp"
#include "threads.hpp"
#include "request.hpp"
#include "future.hpp"
#include "address.hpp"

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

	    /** Communication freezes when I try to access comm from the callbacks
		I have to access it through a pointer, which is initialized for each
		thread inside the constructor.
	    */
	    class communicator<ucx_tag>;
	    static communicator<ucx_tag> *pcomm = NULL;
	    DECLARE_THREAD_PRIVATE(pcomm)

            /** callback_communicator is a class to dispatch send and receive operations to. Each operation can 
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
              * - not thread-safe
              *
              * @tparam Communicator underlying transport communicator
              * @tparam Allocator    allocator type used for allocating shared message buffers */
            template<class Allocator>
	    class callback_communicator<communicator<ucx_tag>, Allocator>: public communicator<ucx_tag>
            {
            public: // member types
                
                using communicator_type = communicator<ucx_tag>;
                using future_type       = typename communicator_type::template future<void>;
                using tag_type          = typename communicator_type::tag_type;
                using rank_type         = typename communicator_type::rank_type;
                using allocator_type    = Allocator;
                using message_type      = shared_message_buffer<allocator_type>;

            private: // member types

            private: // members

                allocator_type      m_alloc;

            public: // ctors

                /** @brief construct from a basic transport communicator
                  * @param comm  the underlying transport communicator
                  * @param alloc the allocator instance to be used for constructing messages */
                callback_communicator(allocator_type alloc = allocator_type{}) 
		    : m_alloc(alloc) {
		    pcomm = this;
		}

                callback_communicator(const callback_communicator&) = delete;
                callback_communicator(callback_communicator&&) = default;
		
                ~callback_communicator() = default;

            public: // send

		/** Send a message to a destination with the given tag. When the message is sent, and
		 * the message ready to be reused, the given call-back is invoked with the destination
		 *  and tag of the message sent.
		 *
		 * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
		 * @tparam CallBack Funciton to call when the message has been sent and the message ready for reuse
		 *
		 * @param msg Const reference to a message to send
		 * @param dst Destination of the message
		 * @param tag Tag associated with the message
		 * @param cb  Call-back function with signature void(int, int)
		 *
		 * @return A value of type `request_type` that can be used to cancel the request if needed.
		 */
		template <typename CallBack>
		void send(const message_type &msg, rank_type dst, tag_type tag, CallBack &&cb)
		{
		    ucp_ep_h ep;
		    ucs_status_ptr_t status;
		    uintptr_t istatus;
		    ucx::ghex_ucx_request_cb<Allocator> *ghex_request;

		    ep = rank_to_ep(dst);

		    CRITICAL_BEGIN(ucp_lock) {

			/* send with callback */
			status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 GHEX_MAKE_SEND_TAG(tag, m_rank), ghex_tag_send_callback<Allocator>);

			// TODO !! C++ doesn't like it..
			istatus = (uintptr_t)status;
			if(UCS_OK == (ucs_status_t)(istatus)){
			    cb(std::move(message_type(msg)), dst, tag);
			} else if(!UCS_PTR_IS_ERR(status)) {
			    ghex_request = (ucx::ghex_ucx_request_cb<Allocator>*)status;
			    
			    /* fill in useful request data */
			    ghex_request->m_peer_rank = dst;
			    ghex_request->m_tag = tag;
			    ghex_request->m_cb = std::forward<CallBack>(cb);
			    ghex_request->m_msg = msg;
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp_lock);
		}


            public: // recieve

		/** Receive a message from a source with the given tag. When the message arrives, and
		 * the message ready to be read, the given call-back is invoked with the source
		 *  and tag of the message sent.
		 *
		 * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
		 * @tparam CallBack Funciton to call when the message has been sent and the message ready to be read
		 *
		 * @param msg Const reference to a message that will contain the data
		 * @param src Source of the message
		 * @param tag Tag associated with the message
		 * @param cb  Call-back function with signature void(int, int)
		 *
		 * @return A value of type `request_type` that can be used to cancel the request if needed.
		 */
		template <typename CallBack>
		void recv(message_type &msg, rank_type src, tag_type tag, CallBack &&cb)
		{
		    ucp_tag_t ucp_tag, ucp_tag_mask;
		    ucs_status_ptr_t status;
		    ucx::ghex_ucx_request_cb<Allocator> *ghex_request;

		    /* set request init data - it might be that the recv completes inside ucp_tag_recv_nb */
		    /* and the callback is called earlier than we initialize the data inside it */

		    // TODO need to lock the worker progress, but this is bad for performance with many threads
		    CRITICAL_BEGIN(ucp_lock) {

			/* sanity check! we could be recursive... OMG! */
			if(early_completion){
			    /* TODO: VERIFY. Error just to catch such situation, if it happens. */
			    /* This should never happen, and even if, should not be a problem: */
			    /* we do not modify anything in the early callback, and the values */
			    /* set here are never used anywhere else. Unless user re-uses the message */
			    /* in his callback after re-submitting a send... Should be told not to. */
			    std::cerr << "recv submitted inside early completion\n";
			}

			early_rank = src;
			early_tag = tag;
			std::function<void(message_type, int, int)> tmpcb = cb;
			early_cb = &tmpcb;  // this is cast to proper type inside the callback, which knows message_type
			early_msg = &msg;
			early_completion = 1;

			GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
			status = ucp_tag_recv_nb(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 ucp_tag, ucp_tag_mask, ghex_tag_recv_callback<Allocator>);

			early_completion = 0;

			if(!UCS_PTR_IS_ERR(status)) {

			    ucs_status_t rstatus;
			    rstatus = ucp_request_check_status (status);
			    if(rstatus != UCS_INPROGRESS){

				ucp_request_free(status);
			    } else {

				ghex_request = (ucx::ghex_ucx_request_cb<Allocator> *)status;

				/* fill in useful request data */
				ghex_request->m_peer_rank = src;
				ghex_request->m_tag = tag;
				ghex_request->m_cb = std::forward<CallBack>(cb);
				ghex_request->m_msg = msg;
			    }
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp_lock);
		}

		/** completion callbacks registered in UCX
		 *  require access to private properties.
		 */
		friend void ghex_tag_recv_callback<Allocator>(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
		friend void ghex_tag_send_callback<Allocator>(void *request, ucs_status_t status);
	    };

	    /** completion callbacks registered in UCX */
	    template <typename Allocator>
	    void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
	    {
		/* 1. extract user callback info from request
		   2. extract message object from request
		   3. decode rank and tag
		   4. call user callback
		   5. release / free the message (ghex is done with it)
		*/
		uint32_t peer_rank = GHEX_GET_SOURCE(info->sender_tag); // should be the same as r->peer_rank
		uint32_t tag = GHEX_GET_TAG(info->sender_tag);          // should be the same as r->tagx

		if(pcomm->early_completion){
		    using MsgType      = shared_message_buffer<Allocator>;

		    /* here we know that the submitting thread is also calling the callback */
		    std::function<void(MsgType, int, int)> *cb =
			static_cast<std::function<void(MsgType, int, int)>*>(pcomm->early_cb);
		    MsgType *tmsg = reinterpret_cast<MsgType *>(pcomm->early_msg);
		    (*cb)(std::move(MsgType(*tmsg)), pcomm->early_rank, pcomm->early_tag);

		    /* do not free the request - it has to be freed after tag_send_nb */
		} else {

		    /* here we know the thrid of the submitting thread, if it is not us */
		    ucx::ghex_ucx_request_cb<Allocator> *r =
			reinterpret_cast<ucx::ghex_ucx_request_cb<Allocator>*>(request);
		    r->m_cb(std::move(r->m_msg), peer_rank, tag);
		    ucp_request_free(request);
		}
	    }

	    template <typename Allocator>
	    void ghex_tag_send_callback(void *request, ucs_status_t status)
	    {
		/* 1. extract user callback info from request
		   2. extract message object from request
		   3. decode rank and tag
		   4. call user callback
		   5. release / free the message (ghex is done with it)
		*/
		ucx::ghex_ucx_request_cb<Allocator> *r =
		    static_cast<ucx::ghex_ucx_request_cb<Allocator>*>(request);
		r->m_cb(std::move(r->m_msg), r->m_peer_rank, r->m_tag);
		ucp_request_free(request);
	    }

	    namespace ucx {

		/** this is used by the request test() function
		    since it has no access to the communicator. 

		    NOTE: has to be ucp_lock'ed by the caller!
		*/
		void worker_progress(){
		    /* TODO: this may not be necessary when critical is no longer used */
		    ucp_worker_progress(pcomm->ucp_worker);
		    if(pcomm->m_nthr > 1){
			ucp_worker_progress(pcomm->ucp_worker);
			ucp_worker_progress(pcomm->ucp_worker);
			ucp_worker_progress(pcomm->ucp_worker);
		    }
		}
	    }
        } // namespace tl
    } // namespace ghex
}// namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_CALLBACK_COMMUNICATOR_HPP */

