/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GHEX_UCX_COMMUNICATOR_HPP
#define GHEX_UCX_COMMUNICATOR_HPP

#include <iostream>
#include <time.h>
#include <map>
#include <functional>

#include <ucp/api/ucp.h>
#include "debug.h"
#include "pmi.h"
#include "locks.hpp"
#include "threads.hpp"

namespace gridtools
{
namespace ghex
{
namespace ucx
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

#define GHEX_MAKE_SEND_TAG(_tag, _dst)			\
    ((((uint64_t) (_tag) ) << GHEX_RANK_BITS)    |	\
     (((uint64_t) (_dst) )))


#define GHEX_MAKE_RECV_TAG(_ucp_tag, _ucp_tag_mask, _tag, _src)		\
    {									\
	_ucp_tag_mask = GHEX_SOURCE_MASK | GHEX_TAG_MASK;		\
	_ucp_tag = ((((uint64_t) (_tag) ) << GHEX_RANK_BITS)    |	\
		    (((uint64_t) (_src) )));				\
    }									\
   
#define GHEX_GET_SOURCE(_ucp_tag)		\
    ((_ucp_tag) & GHEX_SOURCE_MASK)


#define GHEX_GET_TAG(_ucp_tag)			\
    ((_ucp_tag) >> GHEX_RANK_BITS)
  

   
class communicator;

/** Communication freezes when I try to access comm from the callbacks 
    I have to access it through a pointer, which is initialized for each
    thread inside the constructor.
 */
communicator *pcomm;
DECLARE_THREAD_PRIVATE(pcomm)

/** completion callbacks registered in UCX, defined later */
template <typename MsgType>
void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info);
template <typename MsgType>
void ghex_tag_send_callback(void *request, ucs_status_t status);


namespace _impl
{

static std::size_t   ucp_request_size; // size in bytes required for a request by the UCX library
static std::size_t   request_size;     // total request size in bytes (UCX + our data)
       
/** request structure for futures-based comm */
struct ghex_ucx_request {
    ucp_worker_h ucp_worker; // worker thread handling this request
};

/** request structure for callback-based comm */
template<typename MsgType>
struct ghex_ucx_request_cb {
    uint32_t peer_rank;
    uint32_t tag; 
    std::function<void(int, int, MsgType&)> cb;
    MsgType h_msg;
};


void empty_send_cb(void *request, ucs_status_t status)
{
}

/** The future returned by the send and receive
        * operations of a communicator object to check or wait on their status.
        */
struct ucx_future
{
    struct ghex_ucx_request *m_req = NULL;

    ucx_future() = default;
    ucx_future(struct ghex_ucx_request *req) : m_req{req} {}

    // TODO: free the m_req request data when completed!
    /**
       ~ucx_future(){
           free(m_req - _impl::ucp_request_size);
       }
     */

    /** Function to wait until the operation completed */
    void wait()
    {
	if(NULL == m_req) return;
	while (!ready());
    }
    
    /** Function to test if the operation completed
            *
            * @return True if the operation is completed
            */
    bool ready()
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

    /** Cancel the future.
            *
            * @return True if the request was successfully canceled
            */
    bool cancel()
    {
	if(NULL == m_req) return false;
	// ucp_request_cancel(m_req->ucp_worker, m_req);
	// ucp_request_free(m_req);
	
	/* TODO: how to check the status ? what if no callback (nbr) */
        return true;
    }

    private:
    //friend ::gridtools::ghex:ucx::communicator;
    void *request() const { return m_req; }
};

} // namespace _impl

/** Class that provides the functions to send and receive messages. A message
     * is an object with .data() that returns a pointer to `unsigned char`
     * and .size(), with the same behavior of std::vector<unsigned char>.
     * Each message will be sent and received with a tag, bot of type int
     */

class communicator
{
public:
    using future_type = _impl::ucx_future;
    using tag_type = ucp_tag_t;
    using rank_type = int;
    using request_type = _impl::ghex_ucx_request;

    static rank_type m_rank;
    static rank_type m_size;

    rank_type m_thrid;
    rank_type m_nthr;

    static const std::string name;

private:

    static ucp_context_h ucp_context;
    static ucp_worker_h  ucp_worker;

    /** known connection pairs <rank, endpoint address>, 
	created as rquired by the communication pattern
	Has to be per-thread
    */
    std::map<rank_type, ucp_ep_h> connections;

    int early_completion = 0;
    int early_rank;
    int early_tag;
    void *early_cb;
    void *early_msg;

    /* request pool for nbr communications (futures) */
#define REQUEST_POOL_SIZE 10000
    char *ucp_requests = NULL;
    int ucp_request_pos = 0;

public:

    void whoami(){
	printf("I am %d/%d:%d/%d, worker %p\n", m_rank, m_size, m_thrid, m_nthr, ucp_worker);
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
	if(!IN_PARALLEL()) {
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
	m_thrid = 0;
	m_nthr = 1;
	pcomm = this;

	/* only one thread must initialize UCX */
	if(!IN_PARALLEL()) {    
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
		    UCP_PARAM_FIELD_ESTIMATED_NUM_EPS ;

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
		ucp_params.request_size = 64;
		// ucp_params.request_init = _impl::ghex_ucx_request_init;

		/* this should be true if we have per-thread workers 
		   otherwise, if one worker is shared by each thread, it should be false
		   This requires benchmarking. */
		ucp_params.mt_workers_shared = false;

		/* estimated number of end-points - 
		   affects transport selection criteria and theresulting performance */
		ucp_params.estimated_num_eps = m_size;

		/* Mask which specifies particular bits of the tag which can uniquely identify
		   the sender (UCP endpoint) in tagged operations. */
		ucp_params.tag_sender_mask = GHEX_SOURCE_MASK;
	    

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

	    /* ask for UCP request size */
	    {
		ucp_context_attr_t attr = {};
		attr.field_mask = UCP_ATTR_FIELD_REQUEST_SIZE;
		ucp_context_query (ucp_context, &attr);

		/* UCP request size */
		_impl::ucp_request_size = attr.request_size;

		/* Total request size: UCP + GHEX struct*/
		_impl::request_size = attr.request_size + sizeof(struct _impl::ghex_ucx_request);
	    }

	    /* create a worker */
	    {
		memset(&worker_params, 0, sizeof(worker_params));

		/* this should not be used if we have a single worker per thread */
		worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
#ifdef THREAD_MODE_MULTIPLE
		worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
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

	    /* allocate comm request pool */
	    ucp_requests = new char[REQUEST_POOL_SIZE * _impl::request_size];
	    ucp_request_pos = 0;
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
    [[nodiscard]] future_type send(rank_type dst, tag_type tag, MsgType const &msg)
    {
	ucs_status_t status;
	char *request;
	request_type *ghex_request;
	ucp_ep_h ep;
	
	// Dynamic allocation of requests is very slow - need a pool
	// request = (char*)malloc(_impl::request_size);
	request = ucp_requests + _impl::request_size * ucp_request_pos;
	// TODO: check if request is free, if not - look for next one

	ep = rank_to_ep(dst);

	CRITICAL_BEGIN(ucp) {
	    
	    /* send without callback */
	    status = ucp_tag_send_nbr(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
				      GHEX_MAKE_SEND_TAG(tag, m_rank), request + _impl::ucp_request_size);
	
	    if(UCS_OK == status){
	    
		/* send completed immediately */
		ghex_request = nullptr;
	    } else {

		/* update request pool */
		ucp_request_pos++;
		if(ucp_request_pos == REQUEST_POOL_SIZE)
		    ucp_request_pos = 0;

		/* return the future with the request id */
		ghex_request = (request_type*)(request + _impl::ucp_request_size);
		ghex_request->ucp_worker = ucp_worker;
	    }
	} CRITICAL_END(ucp);

	return ghex_request;
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
    void send(rank_type dst, tag_type tag, MsgType &msg, CallBack &&cb)
    {
	ucs_status_ptr_t status;
	uintptr_t istatus;
	_impl::ghex_ucx_request_cb<MsgType> *ghex_request;
	ucp_ep_h ep;

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
		ghex_request = (_impl::ghex_ucx_request_cb<MsgType> *)status;

		/* fill in useful request data */
		ghex_request->peer_rank = dst;
		ghex_request->tag = tag;
		ghex_request->cb = cb;
		ghex_request->h_msg = msg;
	    } else {
		ERR("ucp_tag_send_nb failed");
	    }
	} CRITICAL_END(ucp)
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
    [[nodiscard]] future_type recv(rank_type src, tag_type tag, MsgType &msg) {
	ucs_status_t status;
	char *request;
	request_type *ghex_request;
	ucp_ep_h ep;
	ucp_tag_t ucp_tag, ucp_tag_mask;

	// Dynamic allocation of requests is very slow - need a pool
	// request = (char*)malloc(_impl::request_size);
	request = ucp_requests + _impl::request_size * ucp_request_pos;
	// TODO: check if request is free, if not - look for next one

	ep = rank_to_ep(src);

	CRITICAL_BEGIN(ucp) {

	    /* recv */
	    GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
	    status = ucp_tag_recv_nbr(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
				      ucp_tag, ucp_tag_mask, request + _impl::ucp_request_size);
	    if(UCS_OK != status){
		ERR("ucx recv operation failed");
	    }

	    /* update request pool */
	    ucp_request_pos++;
	    if(ucp_request_pos == REQUEST_POOL_SIZE)
		ucp_request_pos = 0;

	    /* return the future with the request id */
	    ghex_request = (request_type*)(request + _impl::ucp_request_size);
	    ghex_request->ucp_worker = ucp_worker;
	} CRITICAL_END(ucp);

	return ghex_request;
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
	ucs_status_ptr_t status;
	_impl::ghex_ucx_request_cb<MsgType> *ghex_request;
	ucp_tag_t ucp_tag, ucp_tag_mask;

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
	    std::function<void(int, int, MsgType&)> tmpcb = cb;
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

		    // ucp_request_free(status);
		    // return;
		} else {

		    ghex_request = (_impl::ghex_ucx_request_cb<MsgType> *)status;

		    /* fill in useful request data */
		    ghex_request->peer_rank = src;
		    ghex_request->tag = tag;
		    ghex_request->cb = cb;
		    ghex_request->h_msg = msg;
		}
	    } else {
		ERR("ucp_tag_send_nb failed");
	    }
	} CRITICAL_END(ucp)
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
	// TODO: do we need this? where to place this? user code, or here?
	// spinning on progress without any delay is sometimes very slow (?)

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
	flush();

	// TODO: how to assure that all comm is completed before we quit a rank?
	// if we quit too early, we risk infinite waiting on a peer. flush doesn't seem to do the job.
	for(int i=0; i<100000; i++) {
	    ucp_worker_progress(ucp_worker);
	}
    }

    void flush()
    {
	void *request = ucp_worker_flush_nb(ucp_worker, 0, _impl::empty_send_cb);
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

    /** completion callbacks registered in UCX */
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
	std::function<void(int, int, MsgType&)> *cb = static_cast<std::function<void(int, int, MsgType&)>*>(pcomm->early_cb);
	MsgType *msg = static_cast<MsgType *>(pcomm->early_msg);
	(*cb)(pcomm->early_rank, pcomm->early_tag, *msg);
	// ERR("NEEDS TESTING...");

	/* do not free the request - it has to be freed after tag_send_nb */
	/* also do not release the message - it is a pointer to a message owned by the user */
    } else {
	
	/* here we know the thrid of the submitting thread, if it is not us */
	_impl::ghex_ucx_request_cb<MsgType> *r = static_cast<_impl::ghex_ucx_request_cb<MsgType>*>(request);
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
    _impl::ghex_ucx_request_cb<MsgType> *r = static_cast<_impl::ghex_ucx_request_cb<MsgType>*>(request);
    r->cb(r->peer_rank, r->tag, r->h_msg);
    r->h_msg.release();
    ucp_request_free(request);
}

/** this has to be here, because the class needs to be complete */
extern communicator comm;
DECLARE_THREAD_PRIVATE(comm)
communicator comm;

/** static communicator properties, shared between threads */
const std::string communicator::name = "ghex::ucx";
communicator::rank_type communicator::m_rank;
communicator::rank_type communicator::m_size;
ucp_context_h communicator::ucp_context;
ucp_worker_h  communicator::ucp_worker;


} // namespace ucx
} // namespace ghex
} // namespace gridtools

#endif
