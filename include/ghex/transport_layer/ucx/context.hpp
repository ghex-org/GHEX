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
#ifndef INCLUDED_GHEX_TL_UCX_CONTEXT_HPP
#define INCLUDED_GHEX_TL_UCX_CONTEXT_HPP

#include "./communicator.hpp"
#include "../communicator.hpp"

#ifdef GHEX_USE_PMI
// use the PMI interface ...
#include "./address_db_pmi.hpp"
#else
// ... and go to MPI if not available
#include "./address_db_mpi.hpp"
#endif

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename ThreadPrimitives>
            struct transport_context<ucx_tag, ThreadPrimitives>
            {
            public: // member types
                using rank_type         = ucx::endpoint_t::rank_type;
                using worker_type       = ucx::worker_t<ThreadPrimitives>;
                using communicator_type = communicator<ucx::communicator<ThreadPrimitives>>;

            private: // member types

                struct type_erased_address_db_t
                {
                    struct iface
                    {
                        virtual rank_type rank() = 0;
                        virtual rank_type size() = 0;
                        virtual int est_size() = 0;
                        virtual void init(const ucx::address_t&) = 0;
                        virtual ucx::address_t find(rank_type) = 0;
                        virtual ~iface() {}
                    };

                    template<typename Impl>
                    struct impl_t final : public iface
                    {
                        Impl m_impl;
                        impl_t(const Impl& impl) : m_impl{impl} {}
                        impl_t(Impl&& impl) : m_impl{std::move(impl)} {}
                        rank_type rank() override { return m_impl.rank(); }
                        rank_type size() override { return m_impl.size(); }
                        int est_size() override { return m_impl.est_size(); }
                        void init(const ucx::address_t& addr) override { m_impl.init(addr); }
                        ucx::address_t find(rank_type rank) override { return m_impl.find(rank); }
                    };

                    std::unique_ptr<iface> m_impl;

                    template<typename Impl>
                    type_erased_address_db_t(Impl&& impl)
                        : m_impl{std::make_unique<impl_t<std::remove_cv_t<std::remove_reference_t<Impl>>>>(std::forward<Impl>(impl))}{}

                    inline rank_type rank() const { return m_impl->rank(); }
                    inline rank_type size() const { return m_impl->size(); }
                    inline int est_size() const { return m_impl->est_size(); }
                    inline void init(const ucx::address_t& addr) { m_impl->init(addr); }
                    inline ucx::address_t find(rank_type rank) { return m_impl->find(rank); }
                };

                struct ucp_context_h_holder
                {
                    ucp_context_h m_context;
                    ~ucp_context_h_holder() { ucp_cleanup(m_context); }
                };

                using thread_primitives_type = ThreadPrimitives;
                using thread_token          = typename thread_primitives_type::token;
                using worker_vector         = std::vector<std::unique_ptr<worker_type>>;

            private: // members
                
                thread_primitives_type&    m_thread_primitives;
                type_erased_address_db_t   m_db;
                ucp_context_h_holder       m_context;
                std::size_t                m_req_size;
                worker_type                m_worker;  // shared, serialized - per rank
                worker_vector              m_workers; // per thread
                std::vector<thread_token>  m_tokens;

                friend class ucx::worker_t<ThreadPrimitives>;

            public: // static member functions



            public: // ctors
                template<typename DB, typename... Args>
                transport_context(ThreadPrimitives& tp, DB&& db, Args&&...)
                    : m_thread_primitives(tp)
                    , m_db{std::forward<DB>(db)}
                    , m_workers(m_thread_primitives.size())
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
                        UCP_PARAM_FIELD_REQUEST_SIZE      | // size of reserved space in a non-blocking request
                        UCP_PARAM_FIELD_TAG_SENDER_MASK   | // mask which gets sender endpoint from a tag
                        UCP_PARAM_FIELD_MT_WORKERS_SHARED | // multi-threaded context: thread safety
                        UCP_PARAM_FIELD_ESTIMATED_NUM_EPS | // estimated number of endpoints for this context
                        UCP_PARAM_FIELD_REQUEST_INIT      ; // initialize request memory

                    // features
                    context_params.features =
                        UCP_FEATURE_TAG                   ; // tag matching
                    // additional usable request size
                    context_params.request_size = ucx::request_data_size::value;
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
                    context_params.request_init = &ucx::request_init;

                    // initialize UCP
                    GHEX_CHECK_UCX_RESULT(
                        ucp_init(&context_params, config_ptr, &m_context.m_context)
                    );
                    ucp_config_release(config_ptr);

                    // check the actual parameters
                    ucp_context_attr_t attr;
                    attr.field_mask =
                        UCP_ATTR_FIELD_REQUEST_SIZE | // internal request size
                        UCP_ATTR_FIELD_THREAD_MODE;   // thread safety
                    ucp_context_query(m_context.m_context, &attr);
                    m_req_size = attr.request_size;
                    if (attr.thread_mode != UCS_THREAD_MODE_MULTI)
                        throw std::runtime_error("ucx cannot be used with multi-threaded context");

                    // make shared worker
                    // use single-threaded UCX mode, as per developer advice
                    // https://github.com/openucx/ucx/issues/4609
                    m_worker = worker_type(this, &m_thread_primitives, nullptr, UCS_THREAD_MODE_SINGLE);
                    // intialize database
                    m_db.init(m_worker.address());
                }

                communicator_type get_serial_communicator()
                {
                    return {&m_worker,&m_worker};
                }

                communicator_type get_communicator(const thread_token& t)
                {
                    if (!m_workers[t.id()])
                    {
                        m_tokens[t.id()] = t;
                        m_workers[t.id()] = std::make_unique<worker_type>(this, &m_thread_primitives, &m_tokens[t.id()], UCS_THREAD_MODE_SINGLE);
                    }
                    return {&m_worker, m_workers[t.id()].get()};
                }

                rank_type rank() const { return m_db.rank(); }
                rank_type size() const { return m_db.size(); }
                ucp_context_h get() const noexcept { return m_context.m_context; }
            };

            namespace ucx {

                template<typename ThreadPrimitives>
                worker_t<ThreadPrimitives>::worker_t(transport_context_type* c, ThreadPrimitives *tp, thread_token* t, ucs_thread_mode_t mode)
                    : m_context{c}
                    , m_thread_primitives{tp}
                    , m_token_ptr{t}
                    , m_rank(c->rank())
                    , m_size(c->size())
                    {
                        ucp_worker_params_t params;
                        params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
                        params.thread_mode = mode;
                        GHEX_CHECK_UCX_RESULT(
                            ucp_worker_create (c->get(), &params, &m_worker.get())
                        );
                        ucp_address_t* worker_address;
                        std::size_t address_length;
                        GHEX_CHECK_UCX_RESULT(
                            ucp_worker_get_address(m_worker.get(), &worker_address, &address_length)
                        );
                        m_address = address_t{
                            reinterpret_cast<unsigned char*>(worker_address),
                            reinterpret_cast<unsigned char*>(worker_address) + address_length};
                        ucp_worker_release_address(m_worker.get(), worker_address);
                        m_worker.m_moved = false;
                    }

                template<typename ThreadPrimitives>
                const endpoint_t& worker_t<ThreadPrimitives>::connect(rank_type rank)
                {
                    auto it = m_endpoint_cache.find(rank);
                    if (it != m_endpoint_cache.end())
                        return it->second;
                    auto addr = m_context->m_db.find(rank);
                    auto p = m_endpoint_cache.insert(std::make_pair(rank, endpoint_t{rank, m_worker.get(), addr}));
                    return p.first->second;
                }

            } // namespace ucx

            template<class ThreadPrimitives>
            struct context_factory<ucx_tag, ThreadPrimitives>
            {
                static std::unique_ptr<context<ucx_tag, ThreadPrimitives>> create(int num_threads, MPI_Comm comm)
                {
#if defined GHEX_USE_PMI
                    ucx::address_db_pmi addr_db{comm};
#else
                    ucx::address_db_mpi addr_db{comm};
#endif
                    return std::make_unique<context<ucx_tag,ThreadPrimitives>>(num_threads, comm, std::move(addr_db));
                }
            };
            
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_CONTEXT_HPP */
