/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP

#include <iostream>
#include <time.h>
#include <map>
#include <functional>

#include <ucp/api/ucp.h>

#include "../communicator.hpp"
#include "../../common/debug.hpp"

/* TODO: move to a different location */
#include "pmi.h"
#include "locks.hpp"
#include "threads.hpp"
#include "request.hpp"
#include "future.hpp"

namespace gridtools
{
    namespace ghex
    {
	namespace tl
	{

	    /*
	     * GHEX tag structure:
	     *
	     * 01234567 01234567 01234567 01234567 01234567 01234567 01234567 01234567
	     *                                    |
	     *      message tag (32)              |   source rank (32)
	     *                                    |
	     */
#define GHEX_TAG_BITS                       32
#define GHEX_RANK_BITS                      32
#define GHEX_TAG_MASK                       0xffffffff00000000ul
#define GHEX_SOURCE_MASK                    0x00000000fffffffful

#define GHEX_MAKE_SEND_TAG(_tag, _dst)				\
	    ((((uint64_t) (_tag) ) << GHEX_RANK_BITS) |		\
	     (((uint64_t) (_dst) )))


#define GHEX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _src)		\
	    {								\
		_ucp_tag_mask = GHEX_SOURCE_MASK | GHEX_TAG_MASK;	\
		_ucp_tag = ((((uint64_t) (_tag) ) << GHEX_RANK_BITS) |	\
			    (((uint64_t) (_src) )));			\
	    }								\

#define GHEX_GET_SOURCE(_ucp_tag)		\
	    ((_ucp_tag) & GHEX_SOURCE_MASK)


#define GHEX_GET_TAG(_ucp_tag)			\
	    ((_ucp_tag) >> GHEX_RANK_BITS)


	    /** Communication freezes when I try to access comm from the callbacks
		I have to access it through a pointer, which is initialized for each
		thread inside the constructor.
	    */
	    class communicator<ucx_tag>;
	    communicator<ucx_tag> *pcomm = NULL;
	    DECLARE_THREAD_PRIVATE(pcomm)


	    /** completion callbacks registered in UCX, defined later */
	    template <typename MsgType>
	    void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
	    template <typename MsgType>
	    void ghex_tag_send_callback(void *request, ucs_status_t status);


	    /* local definitions - request and future related things */
	    namespace ucx
	    {
		static std::size_t   ucp_request_size; // size in bytes required for a request by the UCX library
		static std::size_t   request_size;     // total request size in bytes (UCX + our data)

		void empty_send_cb(void *request, ucs_status_t status) {}

		void empty_recv_cb(void *request, ucs_status_t status, ucp_tag_recv_info_t *info) {}

		void ghex_request_init_cb(void *request){
		    bzero(request, GHEX_REQUEST_SIZE);
		}
	    }


	    /** Class that provides the functions to send and receive messages. A message
	     * is an object with .data() that returns a pointer to `unsigned char`
	     * and .size(), with the same behavior of std::vector<unsigned char>.
	     * Each message will be sent and received with a tag, bot of type int
	     */

	    template<>
	    class communicator<ucx_tag>
	    {
	    public:
		using tag_type  = ucp_tag_t;
		using rank_type = int;
		using request_type = ucx::request;
		template<typename T>
		using future = ucx::future<T>;

		/* these are static, i.e., shared by threads */
		static rank_type m_rank;
		static rank_type m_size;

		/* these are per-thread */
		rank_type m_thrid;
		rank_type m_nthr;

	    private:

		/* these are static, i.e., shared by threads */
		static ucp_context_h ucp_context;
		static ucp_worker_h  ucp_worker;

		/* these are per-thread */

		/** known connection pairs <rank, endpoint address>,
		    created as rquired by the communication pattern
		    Has to be per-thread
		*/
		std::map<rank_type, ucp_ep_h> connections;

		/* early completion data used in the recv callback */
		int early_completion = 0;
		int early_rank;
		int early_tag;
		void *early_cb;
		void *early_msg;

	    public:

		template<typename MsgType>
		static int get_request_size(){
		    return sizeof(ucx::ghex_ucx_request_cb<MsgType>);
		}
		
		void whoami(){
		    printf("I am Groot! %d/%d:%d/%d, worker %p\n", m_rank, m_size, m_thrid, m_nthr, ucp_worker);
		}

		/*
		  Has to be called at in the begining of the parallel region.
		*/
		void init_mt(){
		    m_thrid = GET_THREAD_NUM();
		    m_nthr = GET_NUM_THREADS();
		    pcomm = this;
		    printf("create communicator %d:%d/%d pointer %p\n", m_rank, m_thrid, m_nthr, pcomm);
		}

		~communicator()
		{
		    if(m_thrid == 0) {
			ucp_worker_flush(ucp_worker);
			/* TODO: this needs to be done correctly. Right now lots of warnings
			   about used / unfreed resources. */
			// ucp_worker_destroy(ucp_worker);
			// ucp_cleanup(ucp_context);
			pmi_finalize();
		    }
		}

		communicator()
		{
		    /* need to set this for single threaded runs */
		    if(!IN_PARALLEL()) {
			m_thrid = 0;
			m_nthr = 1;
			pcomm = this;
		    } else {
			m_thrid = GET_THREAD_NUM();
			m_nthr = GET_NUM_THREADS();
			pcomm = this;
		    }

		    /* only one thread must initialize UCX. 
		       TODO: This should probably be a static method, called once, explicitly, by the user */
		    if(m_thrid == 0) {
			pmi_init();

			// communicator rank and world size
			m_rank = pmi_get_rank();
			m_size = pmi_get_size();

			// UCX initialization
			ucs_status_t status;
			ucp_params_t ucp_params;
			ucp_config_t *config = NULL;
			ucp_worker_params_t worker_params;
			ucp_address_t *worker_address;
			size_t address_length;

			status = ucp_config_read(NULL, NULL, &config);
			if(UCS_OK != status) ERR("ucp_config_read failed");

			/* Initialize UCP */
			{
			    memset(&ucp_params, 0, sizeof(ucp_params));

			    /* pass features, request size, and request init function */
			    ucp_params.field_mask =
				UCP_PARAM_FIELD_FEATURES          |
				UCP_PARAM_FIELD_REQUEST_SIZE      |
				UCP_PARAM_FIELD_TAG_SENDER_MASK   |
				UCP_PARAM_FIELD_MT_WORKERS_SHARED |
				UCP_PARAM_FIELD_ESTIMATED_NUM_EPS |
				UCP_PARAM_FIELD_REQUEST_INIT      ;

			    /* request transport support for tag matching */
			    ucp_params.features =
				UCP_FEATURE_TAG ;

			    // request transport support for wakeup on events
			    // if(use_events){
			    //     ucp_params.features |=
			    // 	UCP_FEATURE_WAKEUP ;
			    // }


			    // TODO: templated request type - how do we know the size??
			    // ucp_params.request_size = sizeof(request_type);
			    ucp_params.request_size = GHEX_REQUEST_SIZE;

			    /* this should be true if we have per-thread workers
			       otherwise, if one worker is shared by all thread, it should be false
			       This requires benchmarking. */
			    ucp_params.mt_workers_shared = false;

			    /* estimated number of end-points -
			       affects transport selection criteria and theresulting performance */
			    ucp_params.estimated_num_eps = m_size;

			    /* Mask which specifies particular bits of the tag which can uniquely identify
			       the sender (UCP endpoint) in tagged operations. */
			    ucp_params.tag_sender_mask = GHEX_SOURCE_MASK;

			    /* Needed to zero the memory region. Otherwise segfaults occured
			       when a std::function destructor was called on an invalid object */
			    ucp_params.request_init = ucx::ghex_request_init_cb;

#if (GHEX_DEBUG_LEVEL == 2)
			    if(0 == pmi_get_rank()){
				LOG("ucp version %s", ucp_get_version_string());
				LOG("ucp features %lx", ucp_params.features);
				ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);
			    }
#endif

			    status = ucp_init(&ucp_params, config, &ucp_context);
			    ucp_config_release(config);

			    if(UCS_OK != status) ERR("ucp_config_init");
			    if(0 == pmi_get_rank()) LOG("UCX initialized");
			}

			/* ask for UCP request size - non-templated version for the futures */
			{
			    ucp_context_attr_t attr = {};
			    attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
			    ucp_context_query (ucp_context, &attr);

			    /* UCP request size */
			    ucx::ucp_request_size = attr.request_size;

			    /* Total request size: UCP + GHEX struct*/
			    ucx::request_size = attr.request_size + sizeof(ucx::ghex_ucx_request);
			}

			/* create a worker */
			{
			    memset(&worker_params, 0, sizeof(worker_params));

			    /* this should not be used if we have a single worker per thread */
			    worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
#ifdef THREAD_MODE_MULTIPLE
			    worker_params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
#else
			    worker_params.thread_mode = UCS_THREAD_MODE_SINGLE;
#endif

			    status = ucp_worker_create (ucp_context, &worker_params, &ucp_worker);
			    if(UCS_OK != status) ERR("ucp_worker_create failed");
			    if(0 == pmi_get_rank()) LOG("UCP worker created");
			}

			/* obtain the worker endpoint address and post it to PMI */
			{
			    status = ucp_worker_get_address(ucp_worker, &worker_address, &address_length);
			    if(UCS_OK != status) ERR("ucp_worker_get_address failed");
			    if(0 == pmi_get_rank()) LOG("UCP worker addres length %zu", address_length);

			    /* update pmi with local address information */
			    pmi_set_string("ghex-rank-address", worker_address, address_length);
			    ucp_worker_release_address(ucp_worker, worker_address);

			    /* invoke global pmi data exchange */
			    // pmi_exchange();
			}
		    }
		}

		rank_type rank() const noexcept { return m_rank; }
		rank_type size() const noexcept { return m_size; }

		ucp_ep_h connect(rank_type rank)
		{
		    ucs_status_t status;
		    ucp_ep_params_t ep_params;
		    ucp_address_t *worker_address;
		    size_t address_length;
		    ucp_ep_h ucp_ep;

		    /* get peer address */
		    pmi_get_string(rank, "ghex-rank-address", (void**)&worker_address, &address_length);

		    /* create endpoint */
		    memset(&ep_params, 0, sizeof(ep_params));
		    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
		    ep_params.address    = worker_address;
		    status = ucp_ep_create (ucp_worker, &ep_params, &ucp_ep);
		    if(UCS_OK != status) ERR("ucp_ep_create failed");
		    free(worker_address);

#if (GHEX_DEBUG_LEVEL == 2)
		    if(0==m_thrid && 0 == pmi_get_rank()){
			ucp_ep_print_info(ucp_ep, stdout);
			ucp_worker_print_info(ucp_worker, stdout);
		    }
#endif

		    LOG("UCP connection established");
		    return ucp_ep;
		}

		ucp_ep_h rank_to_ep(const rank_type &rank)
		{
		    ucp_ep_h ep;

		    /* look for a connection to a given peer
		       create it if it does not yet exist */
		    auto conn = connections.find(rank);
		    if(conn == connections.end()){
			ep = connect(rank);
			connections.emplace(rank, ep);
		    } else {
			/* found an existing connection - return the corresponding endpoint handle */
			ep = conn->second;
		    }

		    return ep;
		}

		/** Send a message to a destination with the given tag.
		 * It returns a future that can be used to check when the message is available
		 * again for the user.
		 *
		 * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
		 *
		 * @param msg Const reference to a message to send
		 * @param dst Destination of the message
		 * @param tag Tag associated with the message
		 *
		 * @return A future that will be ready when the message can be reused (e.g., filled with new data to send)
		 */
		template <typename MsgType>
		[[nodiscard]] future<void> send(rank_type dst, tag_type tag, const MsgType &msg)
		{
		    ucp_ep_h ep;
		    ucs_status_ptr_t status;
		    uintptr_t istatus;
		    char *ucp_request;
		    request_type req;

		    ep = rank_to_ep(dst);

		    CRITICAL_BEGIN(ucp) {

			/* send without callback */
			status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 GHEX_MAKE_SEND_TAG(tag, m_rank), ucx::empty_send_cb);
			
			// TODO !! C++ doesn't like it..
			istatus = (uintptr_t)status;
			if(UCS_OK == (ucs_status_t)(istatus)){

			    /* send completed immediately */
			    req.m_req = nullptr;
			} else if(!UCS_PTR_IS_ERR(status)) {
			    
			    /* store the request */
			    req.m_req = (request_type::req_type)status;
			    req.m_req->ucp_worker = ucp_worker;
			} else {
			    ERR("ucp_tag_recv_nb failed");
			}
		    } CRITICAL_END(ucp);

		    return req;
		}

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
		template <typename MsgType, typename CallBack>
		void send(rank_type dst, tag_type tag, const MsgType &msg, CallBack &&cb)
		{
		    ucp_ep_h ep;
		    ucs_status_ptr_t status;
		    uintptr_t istatus;
		    ucx::ghex_ucx_request_cb<MsgType> *ghex_request;

		    ep = rank_to_ep(dst);

		    CRITICAL_BEGIN(ucp) {

			/* send with callback */
			status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 GHEX_MAKE_SEND_TAG(tag, m_rank), ghex_tag_send_callback<MsgType>);

			// TODO !! C++ doesn't like it..
			istatus = (uintptr_t)status;
			if(UCS_OK == (ucs_status_t)(istatus)){
			    cb(dst, tag, msg);
			} else if(!UCS_PTR_IS_ERR(status)) {
			    ghex_request = (ucx::ghex_ucx_request_cb<MsgType> *)status;
			    
			    /* TODO: investigate */
			    /* OBS: this is needed, otherwise the cb assignment below sometimes segfaults! */
			    /* request_init function seems to solve this problem */
			    // bzero(ghex_request, GHEX_REQUEST_SIZE);
			    
			    /* fill in useful request data */
			    ghex_request->peer_rank = dst;
			    ghex_request->tag = tag;
			    ghex_request->cb = std::forward<CallBack>(cb);
			    ghex_request->h_msg = msg;
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp);
		}


		/** Receive a message from a destination with the given tag.
		 * It returns a future that can be used to check when the message is available
		 * to be read.
		 *
		 * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
		 *
		 * @param msg Const reference to a message that will contain the data
		 * @param src Source of the message
		 * @param tag Tag associated with the message
		 *
		 * @return A future that will be ready when the message can be read
		 */
		template <typename MsgType>
		[[nodiscard]] future<void> recv(rank_type src, tag_type tag, MsgType &msg) {
		    ucp_ep_h ep;
		    ucp_tag_t ucp_tag, ucp_tag_mask;
		    ucs_status_ptr_t status;
		    char *ucp_request;
		    request_type req;

		    CRITICAL_BEGIN(ucp) {

			/* recv */
			GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
			status = ucp_tag_recv_nb(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 ucp_tag, ucp_tag_mask, ucx::empty_recv_cb);

			if(!UCS_PTR_IS_ERR(status)) {

			    ucs_status_t rstatus;
			    rstatus = ucp_request_check_status (status);
			    if(rstatus != UCS_INPROGRESS){
				
				/* recv completed immediately */
				req.m_req = nullptr;
			    } else {

				/* store the request */
				req.m_req = (request_type::req_type)(status);
				req.m_req->ucp_worker = ucp_worker;
			    }
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp);

		    return req;
		}


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
		template <typename MsgType, typename CallBack>
		void recv(rank_type src, tag_type tag, MsgType &msg, CallBack &&cb)
		{
		    ucp_tag_t ucp_tag, ucp_tag_mask;
		    ucs_status_ptr_t status;
		    ucx::ghex_ucx_request_cb<MsgType> *ghex_request;

		    /* set request init data - it might be that the recv completes inside ucp_tag_recv_nb */
		    /* and the callback is called earlier than we initialize the data inside it */

		    // TODO need to lock the worker progress, but this is bad for performance with many threads
		    CRITICAL_BEGIN(ucp) {

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
			std::function<void(int, int, const MsgType&)> tmpcb = cb;
			early_cb = &tmpcb;  // this is cast to proper type inside the callback, which knows MsgType
			early_msg = &msg;
			early_completion = 1;

			/* recv with callback */
			GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
			status = ucp_tag_recv_nb(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 ucp_tag, ucp_tag_mask, ghex_tag_recv_callback<MsgType>);

			early_completion = 0;

			if(!UCS_PTR_IS_ERR(status)) {

			    ucs_status_t rstatus;
			    rstatus = ucp_request_check_status (status);
			    if(rstatus != UCS_INPROGRESS){

				/* early recv completion - callback has been called */

				/*
				  TODO: ask if we need the free. this causes an assertion in UCX, which indicates
				  that for early completion the request is already freed:
				  [c3-4:105532:0:105538] ucp_request.c:76   Assertion `!(flags & UCP_REQUEST_FLAG_RELEASED)' failed
				  0 0x000000000001d370 ucs_fatal_error_message()  ucx-1.6.0/src/ucs/debug/assert.c:36
				  1 0x000000000001d4d6 ucs_fatal_error_format()  ucx-1.6.0/src/ucs/debug/assert.c:52
				  2 0x0000000000016d5d ucp_request_release_common()  ucx-1.6.0/src/ucp/core/ucp_request.c:76
				  3 0x0000000000016d5d ucp_request_free()  ucx-1.6.0/src/ucp/core/ucp_request.c:96
				  4 0x0000000000402fa8 main._omp_fn.0()  GHEX/benchmarks/transport/communicator_ucx.hpp:616
				*/
				
				/* on the other hand, if I don't free this request, then UCX keeps
				   making new requests, so there must be a leak. */
				ucp_request_free(status);
			    } else {

				ghex_request = (ucx::ghex_ucx_request_cb<MsgType> *)status;

				/* TODO: investigate */
				/* OBS: this is needed, otherwise the cb assignment below sometimes segfaults! */
				/* request_init function seems to solve this problem */
				// bzero(ghex_request, GHEX_REQUEST_SIZE);

				/* fill in useful request data */
				ghex_request->peer_rank = src;
				ghex_request->tag = tag;
				ghex_request->cb = std::forward<CallBack>(cb);
				ghex_request->h_msg = msg;
			    }
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp);
		}

		/** Function to invoke to poll the transport layer and check for the completions
		 * of the operations without a future associated to them (that is, they are associated
		 * to a call-back). When an operation completes, the corresponfing call-back is invoked
		 * with the rank and tag associated with that request.
		 *
		 * @return unsigned Non-zero if any communication was progressed, zero otherwise.
		 */
		unsigned progress()
		{
		    int p = 0, i = 0;

		    CRITICAL_BEGIN(ucp) {
			p+= ucp_worker_progress(ucp_worker);
			if(m_nthr>1){
			    p+= ucp_worker_progress(ucp_worker);
			    p+= ucp_worker_progress(ucp_worker);
			    p+= ucp_worker_progress(ucp_worker);
			}
		    } CRITICAL_END(ucp);

		    // the critical section is MUCH better (!!) than the yield
		    if(m_nthr>1) sched_yield();

		    return p;
		}

		void fence()
		{
		    /* this should only be executed by a single thread */
		    THREAD_MASTER () {
			flush();

			// TODO: how to assure that all comm is completed before we quit a rank?
			// if we quit too early, we risk infinite waiting on a peer. flush doesn't seem to do the job.
			for(int i=0; i<100000; i++) {
			    ucp_worker_progress(ucp_worker);
			}
		    }
		    THREAD_BARRIER();
		}

		void barrier()
		{
		    if(m_size > 2) ERR("barrier not implemented for more than 2 ranks");

		    /* this should only be executed by a single thread */
		    THREAD_MASTER () {
			/* do a simple send and recv */
			/* has to be run on master thread thrid 0 */
			future<void> sf, rf;
			rank_type peer_rank = (m_rank+1)%2;
			tag_type tag;
			std::array<int,1> smsg = {1}, rmsg = {0};

			if(m_thrid!=0) ERR("barrier must be run on thread id 0");
			tag = m_nthr-(m_rank+peer_rank+1);

			sf = send(peer_rank, tag, smsg);
			rf = recv(peer_rank, tag, rmsg);
			while(true){
			    if(sf.test() && rf.test()) break;
			    this->progress();
			}
			fprintf(stderr, "barrier done\n");
		    }
		    THREAD_BARRIER();
		}

		void flush()
		{
		    void *request = ucp_worker_flush_nb(ucp_worker, 0, ucx::empty_send_cb);
		    if (request == NULL) {
			return;
		    } else if (UCS_PTR_IS_ERR(request)) {
			ERR("flush failed");
			return;
		    } else {
			ucs_status_t status;
			do {
			    ucp_worker_progress(ucp_worker);
			    status = ucp_request_check_status(request);
			} while (status == UCS_INPROGRESS);
			ucp_request_release(request);
		    }
		}

		/** completion callbacks registered in UCX
		 *  require access to private properties.
		 */
		template <typename MsgType>
		friend void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
		template <typename MsgType>
		friend void ghex_tag_send_callback(void *request, ucs_status_t status);
	    };


	    /** completion callbacks registered in UCX */
	    template <typename MsgType>
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

		    /* here we know that the submitting thread is also calling the callback */
		    std::function<void(int, int, MsgType&)> *cb =
			static_cast<std::function<void(int, int, MsgType&)>*>(pcomm->early_cb);
		    MsgType *msg = static_cast<MsgType *>(pcomm->early_msg);
		    (*cb)(pcomm->early_rank, pcomm->early_tag, *msg);
		    // ERR("NEEDS TESTING...");

		    /* do not free the request - it has to be freed after tag_send_nb */
		    /* also do not release the message - it is a pointer to a message owned by the user */
		} else {

		    /* here we know the thrid of the submitting thread, if it is not us */
		    ucx::ghex_ucx_request_cb<MsgType> *r =
			static_cast<ucx::ghex_ucx_request_cb<MsgType>*>(request);
		    r->cb(peer_rank, tag, r->h_msg);
		    r->h_msg.release();
		    ucp_request_free(request);
		}
	    }

	    template <typename MsgType>
	    void ghex_tag_send_callback(void *request, ucs_status_t status)
	    {
		/* 1. extract user callback info from request
		   2. extract message object from request
		   3. decode rank and tag
		   4. call user callback
		   5. release / free the message (ghex is done with it)
		*/
		ucx::ghex_ucx_request_cb<MsgType> *r =
		    static_cast<ucx::ghex_ucx_request_cb<MsgType>*>(request);
		r->cb(r->peer_rank, r->tag, r->h_msg);
		r->h_msg.release();
		ucp_request_free(request);
	    }

	    /** this has to be here, because the class needs to be complete */
	    // extern communicator<ucx_tag> comm;
	    // DECLARE_THREAD_PRIVATE(comm)
	    // communicator<ucx_tag> comm;

	    /** static communicator properties, shared between threads */
	    communicator<ucx_tag>::rank_type communicator<ucx_tag>::m_rank;
	    communicator<ucx_tag>::rank_type communicator<ucx_tag>::m_size;
	    ucp_context_h communicator<ucx_tag>::ucp_context;
	    ucp_worker_h  communicator<ucx_tag>::ucp_worker;

	} // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP */
