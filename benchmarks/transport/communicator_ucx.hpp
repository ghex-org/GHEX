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
#include <omp.h>
#include <functional>

#include <ucp/api/ucp.h>
#include "debug.h"
#include "pmi.h"

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

namespace _impl
{

static std::size_t   ucp_request_size; // size in bytes required for a request by the UCX library
static std::size_t   request_size;     // total request size in bytes (UCX + our data)
       
/** request structure and init function */
struct ghex_ucx_request {
    ucp_worker_h ucp_worker; // worker thread handling this request
    uint32_t peer_rank;
    uint32_t tag;
    void *cb;                // user-side callback, if any
    void *h_msg;             // user-side message handle
};

/** request structure and init function */
template<typename MsgType, typename CallBack>
struct tghex_ucx_request {
    ucp_worker_h ucp_worker; // worker thread handling this request
    uint32_t peer_rank;
    uint32_t tag; 
    std::function<void(int, int, MsgType&)> cb;
    MsgType h_msg;           // user-side message handle
};

int   rinit_early;
void *rinit_cb;

// template <typename MsgType>
// MsgType &rinit_msg;

/** user-side callback */
/* TODO: a hack, actually MsgType &mesg */
typedef void (*f_callback)(int rank, int tag, void *mesg);

/** completion callbacks registered in UCX */
template <typename MsgType, typename CallBack>
void ghex_tag_recv_callback(void *request, ucs_status_t status, ucp_tag_recv_info_t *info)
{
    /* 1. extract user callback info from request
       2. extract message object from request
       3. decode rank and tag
       4. call user callback
    */
    uint32_t peer_rank = GHEX_GET_SOURCE(info->sender_tag); // should be the same as r->peer_rank
    uint32_t tag = GHEX_GET_TAG(info->sender_tag);          // should be the same as r->tagx

    if(rinit_early){
	ERR("cannot handle early recv yet");
    } else {
	tghex_ucx_request<MsgType, CallBack> *r = static_cast<tghex_ucx_request<MsgType, CallBack>*>(request);
	r->cb(peer_rank, tag, r->h_msg);
	r->h_msg.m_s_message.reset();
	ucp_request_free(request);
    }
}

template <typename MsgType, typename CallBack>
void ghex_tag_send_callback(void *request, ucs_status_t status)
{
    /* 1. extract user callback info from request
       2. extract message object from request
       3. decode rank and tag
       4. call user callback
    */
    tghex_ucx_request<MsgType, CallBack> *r = static_cast<tghex_ucx_request<MsgType, CallBack>*>(request);
    r->cb(r->peer_rank, r->tag, r->h_msg);
    r->h_msg.m_s_message.reset();
    ucp_request_free(request);
}

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
	ucp_worker_progress(m_req->ucp_worker);
	status = ucp_request_check_status(m_req);
	return status != UCS_INPROGRESS;
    }

    /** Cancel the future.
            *
            * @return True if the request was successfully canceled
            */
    bool cancel()
    {
	if(NULL == m_req) return false;
	ucp_request_cancel(m_req->ucp_worker, m_req);
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

    rank_type m_rank;
    rank_type m_size;

    static const std::string name;

private:

    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;

    /** known connection pairs <rank, endpoint address>, 
	created as rquired by the communication pattern
	Has to be per-thread
    */
    std::map<rank_type, ucp_ep_h> connections;

    /* request pool for nbr communications (futures) */
#define REQUEST_POOL_SIZE 10000
    char *ucp_requests;
    int ucp_request_pos = 0;

public:

    ~communicator()
    {
	ucp_worker_flush(ucp_worker);
	/* TODO: this needs to be done correctly. Right now lots of warnings
	   about used / unfreed resources. */
	// ucp_worker_destroy(ucp_worker);
	// ucp_cleanup(ucp_context);
	pmi_finalize();
    }

    communicator()
    {

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
	    // UCP_PARAM_FIELD_REQUEST_INIT      ;

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
	if(0 == pmi_get_rank()){
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
#pragma omp critical(ucp_connection)
	{
	    auto conn = connections.find(rank);
	    if(conn == connections.end()){
		ep = connect(rank);
		connections.emplace(rank, ep);
	    } else {
		/* found an existing connection - return the corresponding endpoint handle */
		ep = conn->second;
	    }
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
    [[nodiscard]] future_type send(MsgType const &msg, rank_type dst, tag_type tag)
    {
	ucs_status_t status;
	char *request;
	ucp_ep_h ep;
	
	// Dynamic allocation of requests is very slow - need a pool
	// request = (char*)malloc(_impl::request_size);
	request = ucp_requests + _impl::request_size * ucp_request_pos;
	// TODO: check if request is free, if not - look for next one

	ep = rank_to_ep(dst);

	/* send without callback */
	status = ucp_tag_send_nbr(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
				  GHEX_MAKE_SEND_TAG(tag, m_rank), request + _impl::ucp_request_size);
	
	if(UCS_OK == status){
	    
	    /* send completed immediately */
	    return nullptr;
	}

	/* update request pool */
	ucp_request_pos++;
	if(ucp_request_pos == REQUEST_POOL_SIZE)
	    ucp_request_pos = 0;

	/* return the future with the request id */
	request_type *ghex_request;
	ghex_request = (request_type*)(request + _impl::ucp_request_size);
	ghex_request->ucp_worker = ucp_worker;
	ghex_request->tag = tag;
	ghex_request->peer_rank = dst;
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
    void send(MsgType &msg, rank_type dst, tag_type tag, CallBack &&cb)
    {
	ucs_status_ptr_t status;
	uintptr_t istatus;
	_impl::tghex_ucx_request<MsgType, CallBack> *ghex_request;
	ucp_ep_h ep;

	ep = rank_to_ep(dst);

	/* send with callback */
	status = ucp_tag_send_nb(ep, msg.data(), msg.size(), ucp_dt_make_contig(1),
				 GHEX_MAKE_SEND_TAG(tag, m_rank), _impl::ghex_tag_send_callback<MsgType, CallBack>);

	// TODO !! C++ doesn't like it..
	istatus = (uintptr_t)status;
	if(UCS_OK == (ucs_status_t)(istatus)){
	    cb(dst, tag, msg);
	} else if(!UCS_PTR_IS_ERR(status)) {
	    ghex_request = (_impl::tghex_ucx_request<MsgType, CallBack> *)status;

	    /* fill in useful request data */
	    ghex_request->peer_rank = dst;
	    ghex_request->tag = tag;
	    ghex_request->cb = std::forward<CallBack>(cb);
	    ghex_request->h_msg = msg;
	} else {
	    ERR("ucp_tag_send_nb failed");
	}
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
    [[nodiscard]] future_type recv(MsgType &msg, rank_type src, tag_type tag) {
	ucs_status_t status;
	char *request;
	ucp_ep_h ep;
	ucp_tag_t ucp_tag, ucp_tag_mask;

	// Dynamic allocation of requests is very slow - need a pool
	// request = (char*)malloc(_impl::request_size);
	request = ucp_requests + _impl::request_size * ucp_request_pos;
	// TODO: check if request is free, if not - look for next one

	ep = rank_to_ep(src);

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
	request_type *ghex_request;
	ghex_request = (request_type*)(request + _impl::ucp_request_size);
	ghex_request->ucp_worker = ucp_worker;
	ghex_request->tag = tag;
	ghex_request->peer_rank = src;
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
    void recv(MsgType &msg, rank_type src, tag_type tag, CallBack &&cb)
    {
	ucs_status_ptr_t status;
	_impl::tghex_ucx_request<MsgType, CallBack> *ghex_request;
	ucp_tag_t ucp_tag, ucp_tag_mask;

	/* set request init data - it might be that the recv completes inside ucp_tag_recv_nb */
	/* and the callback is called earlier than we initialize the data inside it */
	_impl::rinit_early = 1;
	// _impl::rinit_cb = (void*)(&cb);             // TODO !!
	// _impl::rinit_msg = msg;

	/* recv with callback */
	GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
	status = ucp_tag_recv_nb(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1),
				 ucp_tag, ucp_tag_mask, _impl::ghex_tag_recv_callback<MsgType, CallBack>);

	_impl::rinit_early = 0;

	if(!UCS_PTR_IS_ERR(status)) {

	    ucs_status_t rstatus;
	    rstatus = ucp_request_check_status (status);
	    if(rstatus != UCS_INPROGRESS){
		ucp_request_free(status);
		return;
	    }

	    ghex_request = (_impl::tghex_ucx_request<MsgType, CallBack> *)status;

	    /* fill in useful request data */
	    ghex_request->peer_rank = src;
	    ghex_request->tag = tag;
	    ghex_request->cb = std::forward<CallBack>(cb);
	    ghex_request->h_msg = msg;
	} else {
	    ERR("ucp_tag_send_nb failed");
	}
    }

    template <typename CallBack>
    void send(unsigned char *data, size_t size, rank_type dst, tag_type tag, CallBack &&cb)
    {
	ucs_status_ptr_t status;
	uintptr_t istatus;
	_impl::tghex_ucx_request<MsgType, CallBack> *ghex_request;
	ucp_ep_h ep;

	ep = rank_to_ep(dst);

	/* send with callback */
	status = ucp_tag_send_nb(ep, data, size, ucp_dt_make_contig(1),
				 GHEX_MAKE_SEND_TAG(tag, m_rank), _impl::ghex_tag_send_callback<MsgType, CallBack>);

	// TODO !! C++ doesn't like it..
	istatus = (uintptr_t)status;
	if(UCS_OK == (ucs_status_t)(istatus)){
	    // immediate completion. callback? or return value?
	    cb(dst, tag, data);                         // TODO !!
	} else if(!UCS_PTR_IS_ERR(status)) {
	    
	    ghex_request = (request_type*)(status);

	    /* fill in useful request data */
	    ghex_request->peer_rank = dst;
	    ghex_request->tag = tag;
	    ghex_request->cb = (void*)(&cb);            // TODO !!
	    // ghex_request->h_msg = data;
	} else {
	    ERR("ucp_tag_send_nb failed");
	}
    }

    template <typename CallBack>
    void recv(unsigned char *data, size_t size, rank_type src, tag_type tag, CallBack &&cb)
    {
	ucs_status_ptr_t status;
	request_type *ghex_request;
	ucp_tag_t ucp_tag, ucp_tag_mask;

	/* recv with callback */
	GHEX_MAKE_RECV_TAG(ucp_tag, ucp_tag_mask, tag, src);
	status = ucp_tag_recv_nb(ucp_worker, data, size, ucp_dt_make_contig(1),
				 ucp_tag, ucp_tag_mask, _impl::ghex_tag_recv_callback<MsgType, CallBack>);

	if(!UCS_PTR_IS_ERR(status)) {
	    ghex_request = (request_type*)(status);

	    /* fill in useful request data */
	    ghex_request->peer_rank = src;
	    ghex_request->tag = tag;
	    ghex_request->cb = (void*)(&cb);             // TODO !!
	    ghex_request->h_msg = data;
	} else {
	    ERR("ucp_tag_send_nb failed");
	}
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
	// sched_yield();

	return ucp_worker_progress(ucp_worker);
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
};

const std::string communicator::name = "ghex::ucx";

} // namespace ucx
} // namespace ghex
} // namespace gridtools

#endif
