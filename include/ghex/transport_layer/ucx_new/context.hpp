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
            
template<>
struct transport_context<ucx_tag>
{
    struct ucp_context_h_holder
    {
        ucp_context_h m_context;
        ~ucp_context_h_holder() { ucp_cleanup(m_context); }
    };

    using rank_type = ucx::endpoint_t::rank_type;
    using communicator_type = communicator<ucx::communicator>;

private: // members
    const mpi::rank_topology&                m_rank_topology;
    MPI_Comm                                 m_mpi_comm;
    ucx::address_db_t                        m_db;
    ucp_context_h_holder                     m_context;
    std::unique_ptr<ucx::shared_state>       m_shared_state;
    std::unique_ptr<ucx::state>              m_local_state;
    std::vector<std::unique_ptr<ucx::state>> m_local_states;
    std::mutex                               m_mutex;
    
public:
    template<typename DB, typename... Args>
    transport_context(const mpi::rank_topology& t, MPI_Comm mpi_comm_, DB&& db, Args&&...)
    : m_rank_topology{t}
    , m_mpi_comm{mpi_comm_}
    , m_db{std::forward<DB>(db)}
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
        m_shared_state.reset( new ucx::shared_state(m_context.m_context) );
        m_local_state.reset( new ucx::state(m_db, m_context.m_context, m_shared_state.get(), m_rank_topology));

        // intialize database
        m_db.init(m_shared_state->address());
    }

    MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }
                
    communicator_type get_serial_communicator()
    {
        return {m_local_state.get()};
    }

    communicator_type get_communicator()
    {
        std::lock_guard<std::mutex> lk(m_mutex);
        m_local_states.emplace_back(new ucx::state(m_db, m_context.m_context, m_shared_state.get(), m_rank_topology));
        return {m_local_states.back().get()};
    }

    rank_type rank() const { return m_db.rank(); }
    rank_type size() const { return m_db.size(); }
    ucp_context_h get() const noexcept { return m_context.m_context; }
};

struct context_factory<ucx_tag>
{
    static std::unique_ptr<context<ucx_tag>> create(MPI_Comm comm)
    {
        auto new_comm = detail::clone_mpi_comm(comm);
#if defined GHEX_USE_PMI
        ucx::address_db_pmi addr_db{new_comm};
#else
        ucx::address_db_mpi addr_db{new_comm};
#endif
        return std::unique_ptr<context<ucx_tag>>{
            new context<ucx_tag>{new_comm, new_comm, std::move(addr_db)}};
    }
};

} // namespace tl
} // namespace ghex
} // namespace gridtools
