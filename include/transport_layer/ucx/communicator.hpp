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
#include <map>

#include <ucp/api/ucp.h>
#include "debug.h"
#include "pmi.h"

namespace gridtools
{
namespace ghex
{
namespace ucx
{


class communicator;

namespace _impl
{

static std::size_t   ucp_request_size; // size in bytes required for a request by the UCX library
static std::size_t   request_size;     // total request size in bytes (UCX + our data)
    
/** request structure and init function */
struct ghex_ucx_request {
    int rank;
    ucp_worker_h ucp_worker;
};
    
static void ghex_ucx_request_init(void *ptr)
{
    struct ghex_ucx_request *request = (struct ghex_ucx_request *) ptr;
    request->rank = 0;
    request->ucp_worker = nullptr;
}
    
/** The future returned by the send and receive
        * operations of a communicator object to check or wait on their status.
        */
struct ucx_future
{
    struct ghex_ucx_request *m_req;

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
	ucp_request_free(m_req);
	
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
    using request_type = _impl::ghex_ucx_request *;

private:

    rank_type m_rank;
    rank_type m_size;

    ucp_context_h ucp_context;
    ucp_worker_h  ucp_worker;

    /** known connection pairs <rank, endpoint address>, 
	created as rquired by the communication pattern 
    */
    std::map<rank_type, ucp_ep_h> connections;

public:

    communicator() 
    {

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

	memset(&ucp_params, 0, sizeof(ucp_params));
	
	ucp_params.field_mask =
	    UCP_PARAM_FIELD_FEATURES     |
	    UCP_PARAM_FIELD_REQUEST_SIZE |
	    UCP_PARAM_FIELD_REQUEST_INIT ;

	ucp_params.features =
	    UCP_FEATURE_TAG ;

	// support for events
	// if(use_events){
	//     ucp_params.features |=
	// 	UCP_FEATURE_WAKEUP ;
	// }

	ucp_params.request_size = sizeof(request_type);
	ucp_params.request_init = _impl::ghex_ucx_request_init;

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

	{
	    ucp_context_attr_t attr = {};
	    attr.field_mask = 0;
	    attr.field_mask |= UCP_ATTR_FIELD_REQUEST_SIZE;
	    ucp_context_query (ucp_context, &attr);
	    _impl::ucp_request_size = attr.request_size;
	    _impl::request_size = attr.request_size + sizeof(struct _impl::ghex_ucx_request);
	    if(0 == pmi_get_rank()) LOG("ucp_request size %li, struct size %li\n",
					attr.request_size, sizeof(_impl::ghex_ucx_request));
	}

	/* create a worker */
	memset(&worker_params, 0, sizeof(worker_params));
	worker_params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
	worker_params.thread_mode = UCS_THREAD_MODE_MULTI;
	status = ucp_worker_create (ucp_context, &worker_params, &ucp_worker);
	if(UCS_OK != status) ERR("ucp_worker_create failed");
	if(0 == pmi_get_rank()) LOG("UCP worker created");

	/* obtain the worker address */
	status = ucp_worker_get_address(ucp_worker, &worker_address, &address_length);
	if(UCS_OK != status) ERR("ucp_worker_create");
	if(0 == pmi_get_rank()) LOG("UCP worker addres length %zu", address_length);

	/* update pmi with local address information */
	pmi_set_string("ghex-rank-address", worker_address, address_length);
	ucp_worker_release_address(ucp_worker, worker_address);

	/* global pmi data exchange */
	pmi_exchange();
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
	auto conn = connections.find(rank);
	if(conn == connections.end()){
	    ep = connect(rank);
	    connections.emplace(rank, ep);
	    return ep;
	}

	/* found an existing connection */
	return conn->second;
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
	
	request = (char*)malloc(_impl::request_size);
	ep = rank_to_ep(dst);

	/* send without callback */
	status = ucp_tag_send_nbr(ep, msg.data(), msg.size(), ucp_dt_make_contig(1), tag, request + _impl::ucp_request_size);
	
	if(UCS_OK == status){
	    
	    /* send completed immediately */
	    free(request);
	    return nullptr;
	}

	/* return the future with the request id */
	request_type ghex_request;
	ghex_request = (request_type)(request + _impl::ucp_request_size);
	(*ghex_request).rank = dst;
	(*ghex_request).ucp_worker = ucp_worker;
	return ghex_request;
    }

    /** Send a message to a destination with the given tag. This function blocks until the message has been sent and
         * the message ready to be reused
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         */
    template <typename MsgType>
    void blocking_send(MsgType const &msg, rank_type dst, tag_type tag) 
    {
	ucs_status_t status;
	char *request;
	ucp_ep_h ep;
	
	request = (char*)alloca(_impl::request_size);
	ep = rank_to_ep(dst);

	// send
	status = ucp_tag_send_nbr(ep, msg.data(), msg.size(), ucp_dt_make_contig(1), tag, request + _impl::ucp_request_size);

	if (status != UCS_INPROGRESS) {
	    // TODO terminate on error
	    return;
	}

	// wait for completion
	do {
	    ucp_worker_progress(ucp_worker);
	    status = ucp_request_check_status(request + _impl::ucp_request_size);
	} while (status == UCS_INPROGRESS);

	return;
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
	ucp_tag_t tag_mask = 0xffffff;

	request = (char*)malloc(_impl::request_size);
	ep = rank_to_ep(src);

	// recv
	status = ucp_tag_recv_nbr(ucp_worker, msg.data(), msg.size(), ucp_dt_make_contig(1), tag, tag_mask, request + _impl::ucp_request_size);
	if(UCS_OK != status){
	    ERR("ucx recv operation failed");
	}

	// return the future with the request id
	request_type ghex_request;
	ghex_request = (request_type)(request + _impl::ucp_request_size);
	(*ghex_request).rank = src;
	(*ghex_request).ucp_worker = ucp_worker;
	return ghex_request;
    }
};

} // namespace ucx
} // namespace ghex
} // namespace gridtools

#endif
