/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */

#pragma once

#include "../communicator.hpp"
#include "../context.hpp"
#include "./communicator.hpp"

#ifdef GHEX_USE_PMI
// use the PMI interface ...
#include "./address_db_pmi.hpp"
#else
// ... and go to MPI if not available
#include "./address_db_mpi.hpp"
#endif

namespace gridtools{
namespace ghex {
namespace tl {
            
template<typename ThreadPrimitives>
struct transport_context<ucx_tag, ThreadPrimitives>
{
    struct ucp_context_h_holder
    {
        ucp_context_h m_context;
        ~ucp_context_h_holder() { ucp_cleanup(m_context); }
    };

    using thread_primitives_type = ThreadPrimitives;
    using thread_token          = typename thread_primitives_type::token;
    using rank_type = ucx::endpoint_t::rank_type;

    //using communicator_type = ucx::communicator;
    using communicator_type = communicator<ucx::communicator>;

private: // members
    thread_primitives_type&    m_thread_primitives;
    const mpi::rank_topology&  m_rank_topology;
    ucx::address_db_t          m_db;
    ucp_context_h_holder       m_context;
    std::unique_ptr<ucx::shared_state>       m_shared_state;
    std::unique_ptr<ucx::state>              m_local_state;
    std::vector<std::unique_ptr<ucx::state>> m_local_states;
    std::vector<thread_token>  m_tokens;
    
public:
    template<typename DB, typename... Args>
    transport_context(ThreadPrimitives& tp, const mpi::rank_topology& t, DB&& db, Args&&...)
    : m_thread_primitives(tp)
    , m_rank_topology{t}
    , m_db{std::forward<DB>(db)}
    , m_local_states(m_thread_primitives.size())
    , m_tokens(m_thread_primitives.size())
    {
        // read run-time context
        ucp_config_t* config_ptr;
        GHEX_CHECK_UCX_RESULT(
            ucp_config_read(NULL,NULL, &config_ptr)
        );

        // set parameters
        ucp_params_t context_params;
        // define valid fields
        context_params.field_mask =
            UCP_PARAM_FIELD_FEATURES          | // features
            //UCP_PARAM_FIELD_REQUEST_SIZE      | // size of reserved space in a non-blocking request
            UCP_PARAM_FIELD_TAG_SENDER_MASK   | // mask which gets sender endpoint from a tag
            UCP_PARAM_FIELD_MT_WORKERS_SHARED | // multi-threaded context: thread safety
            UCP_PARAM_FIELD_ESTIMATED_NUM_EPS ;//| // estimated number of endpoints for this context
            //UCP_PARAM_FIELD_REQUEST_INIT      ; // initialize request memory

        // features
        context_params.features =
            UCP_FEATURE_TAG                   ; // tag matching
        // additional usable request size
        //context_params.request_size = 16;
        // thread safety
        // this should be true if we have per-thread workers,
        // otherwise, if one worker is shared by all thread, it should be false
        // requires benchmarking.
        context_params.mt_workers_shared = true;
        // estimated number of connections
        // affects transport selection criteria and theresulting performance
        context_params.estimated_num_eps = m_db.est_size();
        // mask
        // mask which specifies particular bits of the tag which can uniquely identify
        // the sender (UCP endpoint) in tagged operations.
        //context_params.tag_sender_mask  = 0x00000000fffffffful;
        context_params.tag_sender_mask  = 0xfffffffffffffffful;
        // needed to zero the memory region. Otherwise segfaults occured
        // when a std::function destructor was called on an invalid object
        //context_params.request_init = &ucx::request_init;

        // initialize UCP
        GHEX_CHECK_UCX_RESULT(
            ucp_init(&context_params, config_ptr, &m_context.m_context)
        );
        ucp_config_release(config_ptr);

        // check the actual parameters
        ucp_context_attr_t attr;
        attr.field_mask = UCP_ATTR_FIELD_THREAD_MODE;   // thread safety
        ucp_context_query(m_context.m_context, &attr);
        if (attr.thread_mode != UCS_THREAD_MODE_MULTI)
            throw std::runtime_error("ucx cannot be used with multi-threaded context");

        // make shared worker
        // use single-threaded UCX mode, as per developer advice
        // https://github.com/openucx/ucx/issues/4609
        m_shared_state.reset( new ucx::shared_state(m_context.m_context, m_tokens.size()) );
        m_local_state.reset( new ucx::state(m_db, m_context.m_context, m_shared_state.get(), m_rank_topology));

        // intialize database
        m_db.init(m_shared_state->address());
    }
                
    communicator_type get_serial_communicator()
    {
        return {m_local_state.get()};
    }

    communicator_type get_communicator(const thread_token& t)
    {
        if (!m_local_states[t.id()])
        {
            m_tokens[t.id()] = t;
            m_local_states[t.id()].reset(new ucx::state(m_db, m_context.m_context, m_shared_state.get(), m_rank_topology));
        }
        return {m_local_states[t.id()].get()};
    }

    rank_type rank() const { return m_db.rank(); }
    rank_type size() const { return m_db.size(); }
    ucp_context_h get() const noexcept { return m_context.m_context; }
};

template<class ThreadPrimitives>
struct context_factory<ucx_tag, ThreadPrimitives>
{
    static std::unique_ptr<context<ucx_tag, ThreadPrimitives>> create(int num_threads, MPI_Comm comm)
    {
        auto new_comm = detail::clone_mpi_comm(comm);
#if defined GHEX_USE_PMI
        ucx::address_db_pmi addr_db{new_comm};
#else
        ucx::address_db_mpi addr_db{new_comm};
#endif
        return std::unique_ptr<context<ucx_tag, ThreadPrimitives>>{
            new context<ucx_tag,ThreadPrimitives>{num_threads, new_comm, std::move(addr_db)}};
    }
};

} // namespace tl
} // namespace ghex
} // namespace gridtools
