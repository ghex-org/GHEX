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

#include <vector>
#include <memory>
#include "../context.hpp"
#include "./worker.hpp"
#include "./communicator.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace ucx {

                struct context_t
                {
                public: // member types
                    
                    using rank_type         = typename endpoint_t::rank_type;
                    using communicator_type = ::gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;

                private: // member types

                    struct type_erased_address_db_t
                    {
                        struct iface
                        {
                            virtual rank_type rank() = 0;
                            virtual rank_type size() = 0;
                            virtual int est_size() = 0;
                            virtual void init(const address_t&) = 0;
                            virtual address_t* find(rank_type) = 0;
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
                            void init(const address_t& addr) override { m_impl.init(addr); }
                            address_t* find(rank_type rank) override { return m_impl.find(rank); }
                        };

                        std::unique_ptr<iface> m_impl;

                        template<typename Impl>
                        type_erased_address_db_t(Impl&& impl)
                        : m_impl{
                            std::make_unique<
                                impl_t<
                                    typename std::remove_cv<
                                        typename std::remove_reference<
                                            Impl
                                        >::type
                                    >::type
                                >
                            >(std::forward<Impl>(impl))}
                        {}

                        inline rank_type rank() const { return m_impl->rank(); }
                        inline rank_type size() const { return m_impl->size(); }
                        inline int est_size() const { return m_impl->est_size(); }
                        inline void init(const address_t& addr) { m_impl->init(addr); }
                        inline address_t* find(rank_type rank) { return m_impl->find(rank); }
                    };
                    
                    using worker_vector = std::vector<std::unique_ptr<worker_t>>;

                    struct ucp_context_h_holder
                    {
                        ucp_context_h m_context;
                        ~ucp_context_h_holder()
                        {
                            ucp_cleanup(m_context);
                        }
                    };

                private: // members

                    type_erased_address_db_t  m_db;
                    ucp_context_h_holder      m_context;
                    std::size_t               m_req_size;
                    worker_t                  m_worker;
                    worker_vector             m_workers;
                    worker_vector             m_workers_ts;

                public: // ctors

                    template<typename DB>
                    context_t(DB&& db)
                    : m_db{std::forward<DB>(db)}
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
                            UCP_PARAM_FIELD_ESTIMATED_NUM_EPS ; // estimated number of endpoints for this context

                        // features
                        context_params.features =
                            UCP_FEATURE_TAG                   ; // tag matching
                        // request size
                        //context_params.request_size = 64;
                        context_params.request_size = 0;
                        // thread safety
                        context_params.mt_workers_shared = 1;
                        // estimated number of connections
                        context_params.estimated_num_eps = m_db.est_size();
                        // mask
                        //context_params.tag_sender_mask  = 0x00000000fffffffful;
                        context_params.tag_sender_mask  = 0xfffffffffffffffful;

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

                        m_worker = worker_t(this, 0u, true);
                        m_db.init(m_worker.address());
                    }
                    
                    context_t(const context_t&) = delete;
                    context_t(context_t&& other) noexcept = delete;

                public: // member functions

                    rank_type rank() const { return m_db.rank(); }
                    rank_type size() const { return m_db.size(); }
                    ucp_context_h get() const noexcept { return m_context.m_context; }
                    auto& db() noexcept { return m_db; }
                    
                    communicator_type make_communicator()
                    {
                        const std::size_t index = m_workers.size()+1u;
                        m_workers.push_back(std::make_unique<worker_t>(this,index,false));
                        m_workers_ts.push_back(std::make_unique<worker_t>(this,index,true));
                        return {m_workers.back().get(), m_workers_ts.back().get(), &m_worker}; 
                    }
                };

                // worker implementation

                worker_t::worker_t(context_t* context, std::size_t index, bool shared)
                : m_context(context)
                , m_index(index)
                , m_shared(shared)
                , m_rank(context->rank())
                , m_size(context->size())
                , m_mutex(std::make_unique<mutex_type>())
                {
                    ucp_worker_params_t params;
                    params.field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE;
                    if (shared)
                    //  params.thread_mode = UCS_THREAD_MODE_MULTI;
                        params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
                    else
                        //params.thread_mode = UCS_THREAD_MODE_SERIALIZED;
                        params.thread_mode = UCS_THREAD_MODE_SINGLE;
                    GHEX_CHECK_UCX_RESULT(
                        ucp_worker_create (context->get(), &params, &m_worker.get())
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
                }
                
                const endpoint_t& worker_t::connect(rank_type rank)
                {
                    auto it = m_endpoint_cache.find(rank);
                    if (it != m_endpoint_cache.end())
                        return it->second;
                    if (auto addr_ptr = m_context->db().find(rank))
                    {
                        auto p = m_endpoint_cache.insert(std::make_pair(rank, endpoint_t{rank, m_worker.get(), *addr_ptr}));
                        return p.first->second;
                    }
                    else
                        throw std::runtime_error("could not connect to endpoint");
                }

            } // namespace ucx
            
            template<>
            struct context<ucx_tag>
            {
                //using impl_type = ucx::context_t;
                //using rank_type         = typename impl_type::rank_type;
                //using communicator_type = typename impl_type::communicator_type;

                //std::unique_ptr<impl_type> m_impl;

                template<typename DB>
                context(DB&& db)
                //: m_impl{ std::make_unique<impl_type>( std::forward<DB>(db) ) }
                {}

                //communicator_type make_communicator() { return m_impl->make_communicator(); }

                //rank_type rank() { return m_impl->rank(); }

                //rank_type size() { return m_impl->size(); }
            }; 

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_CONTEXT_HPP */

