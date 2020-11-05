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
#ifndef INCLUDED_GHEX_TL_UCX_CONTEXT_HPP
#define INCLUDED_GHEX_TL_UCX_CONTEXT_HPP

#include "../context.hpp"
#include "./communicator.hpp"
#include "../communicator.hpp"
#include "../util/pthread_spin_mutex.hpp"

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

            template <>
            struct transport_context<ucx_tag>
            {
            public: // member types
                using rank_type         = ucx::endpoint_t::rank_type;
                using worker_type       = ucx::worker_t;
                using communicator_type = communicator<ucx::communicator>;

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

                using worker_vector         = std::vector<std::unique_ptr<worker_type>>;

            private: // members

                using mutex_t = pthread_spin::recursive_mutex;
                MPI_Comm m_mpi_comm;
                const mpi::rank_topology&  m_rank_topology;
                type_erased_address_db_t   m_db;
                ucp_context_h_holder       m_context;
                std::size_t                m_req_size;
                worker_type                m_worker;  // shared, serialized - per rank
                worker_vector              m_workers; // per thread
                mutex_t                    m_mutex;

                friend class ucx::worker_t;

            public: // static member functions



            public: // ctors
                template<typename DB, typename... Args>
                transport_context(const mpi::rank_topology& t, MPI_Comm comm, DB&& db, Args&&...)
                    : m_mpi_comm{comm}
                    , m_rank_topology{t}
                    , m_db{std::forward<DB>(db)}
                    , m_workers()
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
                    m_worker = worker_type{this, m_mutex, UCS_THREAD_MODE_SINGLE};

                    // intialize database
                    m_db.init(m_worker.address());
                }

                MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }

                communicator_type get_serial_communicator()
                {
                    return {&m_worker,&m_worker};
                }

                communicator_type get_communicator()
                {
                    std::lock_guard<mutex_t> lock(m_mutex); // we need to guard only the isertion in the vector, but this is not a performance critical section
                    m_workers.push_back(std::make_unique<worker_type>(this, m_mutex, UCS_THREAD_MODE_SERIALIZED));
                return {&m_worker, m_workers[m_workers.size()-1].get()};
                }

                rank_type rank() const { return m_db.rank(); }
                rank_type size() const { return m_db.size(); }
                ucp_context_h get() const noexcept { return m_context.m_context; }
            };

            namespace ucx {

                inline worker_t::worker_t(transport_context_type* c, mutex_t& mm, ucs_thread_mode_t mode)
                    : m_context{c}
                    , m_rank(c->rank())
                    , m_size(c->size())
                    , m_mutex_ptr{&mm}
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

                inline const endpoint_t& worker_t::connect(rank_type rank)
                {
                    auto it = m_endpoint_cache.find(rank);
                    if (it != m_endpoint_cache.end())
                        return it->second;
                    auto addr = m_context->m_db.find(rank);
                    auto p = m_endpoint_cache.insert(std::make_pair(rank, endpoint_t{rank, m_worker.get(), addr}));
                    return p.first->second;
                }

                inline const ::gridtools::ghex::tl::mpi::rank_topology& worker_t::rank_topology() const noexcept { return (*m_context).m_rank_topology; }

            } // namespace ucx

            template<>
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
#endif /* INCLUDED_GHEX_TL_UCX_CONTEXT_HPP */
