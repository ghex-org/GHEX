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
#include <deque>

#include <ucp/api/ucp.h>

#include "../communicator.hpp"
#include "../../common/debug.hpp"

#ifdef USE_PMIX
#define USE_PMI
#include "../util/pmi/pmix/pmi.hpp"
using PmiType = gridtools::ghex::tl::pmi<gridtools::ghex::tl::pmix_tag>;
#endif

#include "ucp_lock.hpp"
#include "request.hpp"
#include "future.hpp"
#include "address.hpp"

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

	    class communicator<ucx_tag>;
	    namespace ucx 
	    {
		static communicator<ucx_tag> *pcomm = NULL;
		DECLARE_THREAD_PRIVATE(pcomm)
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
		using size_type = int;
		using request_type = ucx::request;
		template<typename T>
		using future = ucx::future<T>;
                using address_type   = ucx::address;
                using traits         = int;

		/* these are static, i.e., shared by threads */
		static rank_type m_rank;
		static rank_type m_size;

		/* these are per-thread */
		thread_rank_type m_thrid;
		thread_rank_type m_nthr;

	    private:

		/* these are static, i.e., shared by threads */
		static ucp_context_h ucp_context;
		static ucp_worker_h  ucp_worker;

		/* these are per-thread */

#ifdef USE_PMI
		/** PMI interface to obtain peer addresses */
		PmiType pmi_impl;
#endif
		/** known connection pairs <rank, endpoint address>,
		    created as rquired by the communication pattern
		    Has to be per-thread
		*/
		std::map<rank_type, ucp_ep_h> connections;

	    public:

		template<typename MsgType>
		static int get_request_size(){
		    return sizeof(ucx::ghex_ucx_request_cb<MsgType>);
		}
		
		void whoami(){
		    printf("I am Groot! %d/%d:%d/%d, worker %p\n", m_rank, m_size, m_thrid, m_nthr, ucp_worker);
		}
		
                rank_type rank() const noexcept { return m_rank; }
                size_type size() const noexcept { return m_size; }

		~communicator()
		{
		    THREAD_MASTER (){
			ucp_worker_flush(ucp_worker);
			// ucp_worker_destroy(ucp_worker);
			// ucp_cleanup(ucp_context);
		    }
		}

		communicator(const traits& t = traits{})
		{
		    /* need to set this for single threaded runs */
		    m_thrid = GET_THREAD_NUM();
		    m_nthr = GET_NUM_THREADS();
		    ucx::pcomm = this;

		    /* only one thread must initialize UCX. 
		       TODO: This should probably be a static method, called once, explicitly, by the user */
		    THREAD_MASTER (){

#ifdef USE_PMI
			// communicator rank and world size
			m_rank = pmi_impl.rank();
			m_size = pmi_impl.size();
#endif
			
#ifdef THREAD_MODE_MULTIPLE
#ifndef USE_OPENMP_LOCKS
			LOCK_INIT(ucp_lock);
#endif
#endif

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
			    if(0 == m_rank){
			    	LOG("ucp version %s", ucp_get_version_string());
			    	LOG("ucp features %lx", ucp_params.features);
			    	ucp_config_print(config, stdout, NULL, UCS_CONFIG_PRINT_CONFIG);
			    }
#endif

			    status = ucp_init(&ucp_params, config, &ucp_context);
			    ucp_config_release(config);

			    if(UCS_OK != status) ERR("ucp_config_init");
			    if(0 == m_rank) LOG("UCX initialized");
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
			    if(0 == m_rank) LOG("UCP worker created");
			}

#ifdef USE_PMI
			/* obtain the worker endpoint address and post it to PMI */
			{
			    status = ucp_worker_get_address(ucp_worker, &worker_address, &address_length);
			    if(UCS_OK != status) ERR("ucp_worker_get_address failed");
			    if(0 == m_rank) LOG("UCP worker addres length %zu", address_length);

			    /* update pmi with local address information */
			    std::vector<char> data((const char*)worker_address, (const char*)worker_address + address_length);
			    pmi_impl.set("ghex-rank-address", data);
			    ucp_worker_release_address(ucp_worker, worker_address);

			    /* invoke global pmi data exchange */
			    // pmi_exchange();
			}
#endif
		    }
		}

		address_type address(){
		    ucs_status_t status;
		    ucp_address_t *worker_address;
		    size_t address_length;

		    status = ucp_worker_get_address(ucp_worker, &worker_address, &address_length);
		    if(UCS_OK != status) ERR("ucp_worker_get_address failed");
		    return address_type(ucp_worker, worker_address, address_length);
		}

		ucp_ep_h connect(address_type worker_address)
		{
		    return connect(worker_address.m_addr);
		}

		ucp_ep_h connect(ucp_address_t *worker_address)
		{
		    ucs_status_t status;
		    ucp_ep_params_t ep_params;
		    ucp_ep_h ucp_ep;

		    /* create endpoint */
		    memset(&ep_params, 0, sizeof(ep_params));
		    ep_params.field_mask = UCP_EP_PARAM_FIELD_REMOTE_ADDRESS;
		    ep_params.address    = worker_address;
		    status = ucp_ep_create (ucp_worker, &ep_params, &ucp_ep);
		    if(UCS_OK != status) ERR("ucp_ep_create failed");

#if (GHEX_DEBUG_LEVEL == 2)
		    if(0 == m_thrid && 0 == m_rank){
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

			ucp_address_t *worker_address;
#ifdef USE_PMI
			/* get peer address - we have ownership of the address */
			std::vector<char> data = pmi_impl.get_bytes(rank, "ghex-rank-address");
			worker_address = (ucp_address_t*)data.data();
#else
			ERR("PMI is not enabled. Don't know how to obtain peer address.");
#endif

			ep = connect(worker_address);
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
		[[nodiscard]] future<void> send(const MsgType &msg, rank_type dst, tag_type tag)
		{
		    ucp_ep_h ep;
		    ucs_status_ptr_t status;
		    uintptr_t istatus;
		    char *ucp_request;
		    request_type req;

		    ep = rank_to_ep(dst);

		    CRITICAL_BEGIN(ucp_lock) {

			/* send */
			status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
						 GHEX_MAKE_SEND_TAG(tag, m_rank), ucx::empty_send_cb);
			
			// TODO !! C++ doesn't like it..
			istatus = (uintptr_t)status;
			if(UCS_OK == (ucs_status_t)(istatus)){

			    /* send completed immediately */
			    req.m_req = nullptr;
			} else if(!UCS_PTR_IS_ERR(status)) {
			    
			    /* return the request */
			    req.m_req = (request_type::req_type)(status);
			} else {
			    ERR("ucp_tag_recv_nb failed");
			}
		    } CRITICAL_END(ucp_lock);

		    return req;
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
		[[nodiscard]] future<void> recv(MsgType &msg, rank_type src, tag_type tag) {
		    ucp_ep_h ep;
		    ucp_tag_t ucp_tag, ucp_tag_mask;
		    ucs_status_ptr_t status;
		    char *ucp_request;
		    request_type req;

		    CRITICAL_BEGIN(ucp_lock) {

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

				/* return the request */
				req.m_req = (request_type::req_type)(status);
			    }
			} else {
			    ERR("ucp_tag_send_nb failed");
			}
		    } CRITICAL_END(ucp_lock);

		    return req;
		}


		/** Function to invoke to poll the transport layer and check for the completions
		 * of the operations without a future associated to them (that is, they are associated
		 * to a call-back). When an operation completes, the corresponfing call-back is invoked
		 * with the rank and tag associated with that request.
		 *
		 * @return unsigned Non-zero if any communication was progressed, zero otherwise.
		 */
		// #include <time.h>
		// struct timespec progress_time;
		// int time_init = 0;
		// int tthreashold = 0;
		// int empty_progress = 0;
		unsigned progress()
		{
		    int p = 0, i = 0;

		    // struct timespec time;
		    // if(tthreashold){
		    // 	if(!time_init){
		    // 	    clock_gettime(CLOCK_REALTIME, &progress_time);
		    // 	    time_init = 1;
		    // 	} else {
		    // 	    long tdiff;
		    // 	    clock_gettime(CLOCK_REALTIME, &time);
		    // 	    tdiff = (time.tv_sec-progress_time.tv_sec)*1000000000 + (time.tv_nsec-progress_time.tv_nsec);
		    // 	    if(tdiff<tthreashold) {
		    // 		return 0;
		    // 	    }
		    // 	    progress_time = time;
		    // 	}
		    // }

		    CRITICAL_BEGIN(ucp_lock) {
			p+= ucp_worker_progress(ucp_worker);

			/* the below improves one-directional tests, but substantially
			   slows down the bi-directional ones!
			*/
			// if(m_nthr>1){
			//     /* TODO: this may not be necessary when critical is no longer used */
			//     p+= ucp_worker_progress(ucp_worker);
			//     p+= ucp_worker_progress(ucp_worker);
			//     p+= ucp_worker_progress(ucp_worker);
			// }
		    } CRITICAL_END(ucp_lock);

		    // the below is necessary when using spin-locks
		    // NOTE: DO NOT USE! very strange effects in MT
		    // send/receive is split into phases, e.g., first send, then receive on a given rank. WTF?
		    // if(m_nthr>1) sched_yield();

		    return p;
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

			// TODO: use something that cannot be used by the user. otherwise trouble...
			tag = 0x800000;

			sf = send(smsg, peer_rank, tag);
			rf = recv(rmsg, peer_rank, tag);
			while(true){
			    if(sf.test() && rf.test()) break;
			    progress();
			}
		    }
		    THREAD_BARRIER();
		}

		/* this must only be executed by a single thread */
		void flush()
		{
		    /* this should only be executed by a single thread */
		    THREAD_MASTER () {
			void *request = ucp_worker_flush_nb(ucp_worker, 0, ucx::empty_send_cb);
			if (request == NULL) {
			    /* done */
			} else if (UCS_PTR_IS_ERR(request)) {
			    /* terminate */
			    ERR("flush failed");
			} else {
			    ucs_status_t status;
			    do {
				ucp_worker_progress(ucp_worker);
				status = ucp_request_check_status(request);
			    } while (status == UCS_INPROGRESS);
			    ucp_request_release(request);
			}
		    }
		    THREAD_BARRIER();
		}

		friend void ucx::worker_progress();
	    };

	    /** static communicator properties, shared between threads */
	    communicator<ucx_tag>::rank_type communicator<ucx_tag>::m_rank;
	    communicator<ucx_tag>::rank_type communicator<ucx_tag>::m_size;
	    ucp_context_h communicator<ucx_tag>::ucp_context = 0;
	    ucp_worker_h  communicator<ucx_tag>::ucp_worker = 0;

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
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_COMMUNICATOR_HPP */
