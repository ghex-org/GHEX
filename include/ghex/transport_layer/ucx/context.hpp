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
#ifndef INCLUDED_TL_MPI_CONTEXT_HPP
#define INCLUDED_TL_MPI_CONTEXT_HPP

//#include "../context.hpp"
////#include "./communicator.hpp"
//#include "../ucx3/worker.hpp"
////#include "../common/moved_bit.hpp"
//#include "../../threads/atomic/primitives.hpp"
//#include "./request2.hpp"
#include "./future2.hpp"
#include <iostream>

#define GHEX_NO_EXCEPTIONS

namespace gridtools {
    namespace ghex {
        namespace tl {
		    
            namespace ucx {    

                template<typename ThreadPrimitives>
                struct communicator
                {
                    using worker_type            = worker_t<ThreadPrimitives>;
                    using parallel_context_type  = parallel_context<ThreadPrimitives>;
                    using thread_token           = typename parallel_context_type::thread_token;
                    using rank_type              = endpoint_t::rank_type;
                    using tag_type               = typename worker_type::tag_type;
                    using request                = request_ft<ThreadPrimitives>;
                    template<typename T>
                    using future                 = future_t<T,ThreadPrimitives>;
                    // needed for now for high-level API
                    using address_type           = rank_type;
                    
                    using request_cb_type        = request_cb<ThreadPrimitives>;
                    using request_cb_data_type   = typename request_cb_type::data_type;
                    using request_cb_state_type  = typename request_cb_type::state_type;
                    using message_type           = typename request_cb_type::message_type;

                    worker_type*  m_recv_worker;
                    worker_type*  m_send_worker;
                    ucp_worker_h  m_ucp_rw;
                    ucp_worker_h  m_ucp_sw;
                    rank_type     m_rank;
                    rank_type     m_size;

                    communicator(worker_type* rw, worker_type* sw) noexcept
                    : m_recv_worker{rw}
                    , m_send_worker{sw}
                    , m_ucp_rw{rw->get()}
                    , m_ucp_sw{sw->get()}
                    , m_rank{m_send_worker->rank()}
                    , m_size{m_send_worker->size()}
                    {}

                    communicator(const communicator&) = default;
                    communicator(communicator&&) = default;
                    communicator& operator=(const communicator&) = default;
                    communicator& operator=(communicator&&) = default;

                    //rank_type rank() const noexcept { return m_send_worker->rank(); }
                    //rank_type size() const noexcept { return m_send_worker->size(); }
                    rank_type rank() const noexcept { return m_rank; }
                    rank_type size() const noexcept { return m_size; }
                    address_type address() const { return rank(); }

                    static void empty_send_callback(void *, ucs_status_t) {}
                    static void empty_recv_callback(void *, ucs_status_t, ucp_tag_recv_info_t*) {}

                    template <typename MsgType>
                    [[nodiscard]] future<void> send(const MsgType &msg, rank_type dst, tag_type tag)
                    {
                        const auto& ep = m_send_worker->connect(dst);
                        const auto stag = ((std::uint_fast64_t)tag << 32) | 
                                           (std::uint_fast64_t)(rank());
                        auto ret = ucp_tag_send_nb(
                            ep.get(),                                        // destination
                            msg.data(),                                      // buffer
                            msg.size()*sizeof(typename MsgType::value_type), // buffer size
                            ucp_dt_make_contig(1),                           // data type
                            stag,                                            // tag
                            &communicator::empty_send_callback);             // callback function pointer: empty here
                        
                        if (reinterpret_cast<std::uintptr_t>(ret) == UCS_OK)
                        {
                            // send operation is completed immediately and the call-back function is not invoked
                            return request{nullptr};
                        } 
#ifndef GHEX_NO_EXCEPTIONS
                        else if(!UCS_PTR_IS_ERR(ret))
#endif
                        {
                            return request{request::data_type::construct(ret, m_recv_worker, m_send_worker, request_kind::send)};
                        }
#ifndef GHEX_NO_EXCEPTIONS
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - send operation failed");
                        }
#endif
                    }
		
                    template <typename MsgType>
                    [[nodiscard]] future<void> recv(MsgType &msg, rank_type src, tag_type tag)
                    {
                        const auto rtag = ((std::uint_fast64_t)tag << 32) | 
                                           (std::uint_fast64_t)(src);
                        return m_send_worker->m_parallel_context->thread_primitives().critical(
                            [this,rtag,&msg,src,tag]()
                            {
                                auto ret = ucp_tag_recv_nb(
                                    m_recv_worker->get(),                            // worker
                                    msg.data(),                                      // buffer
                                    msg.size()*sizeof(typename MsgType::value_type), // buffer size
                                    ucp_dt_make_contig(1),                           // data type
                                    rtag,                                            // tag
                                    ~std::uint_fast64_t(0ul),                        // tag mask
                                    &communicator::empty_recv_callback);             // callback function pointer: empty here
#ifndef GHEX_NO_EXCEPTIONS
                                if(!UCS_PTR_IS_ERR(ret))
#endif
                                {
			                        if (UCS_INPROGRESS != ucp_request_check_status(ret))
                                    {
				                        // recv completed immediately
		    		                    // we need to free the request here, not in the callback
                                        auto ucx_ptr = ret;
                                        request_init(ucx_ptr);
				                        ucp_request_free(ucx_ptr);
                                        return request{nullptr};
                                    }
                                    else
                                    {
                                        return request{request::data_type::construct(ret, m_recv_worker, m_send_worker, request_kind::recv)};
                                    }
                                }
#ifndef GHEX_NO_EXCEPTIONS
                                else
                                {
                                    // an error occurred
                                    throw std::runtime_error("ghex: ucx error - recv operation failed");
                                }
#endif
                            }
                        );
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
                        int p = 0;
                        p+= ucp_worker_progress(m_ucp_sw);
                        p+= ucp_worker_progress(m_ucp_sw);
                        p+= ucp_worker_progress(m_ucp_sw);
                        //using tp_t=std::remove_reference_t<decltype(m_send_worker->m_parallel_context->thread_primitives())>;
                        //using lk_t=typename tp_t::lock_type;
                        //lk_t lk(m_send_worker->m_parallel_context->thread_primitives().m_mutex);
                        m_send_worker->m_parallel_context->thread_primitives().critical(
                            [this,&p]()
                            {
                                p+= ucp_worker_progress(m_ucp_rw);
                                p+= ucp_worker_progress(m_ucp_rw);
                            }
                        );
                        return p;
                    }

                    template<typename V>
                    using ref_message = ::gridtools::ghex::tl::cb::ref_message<V>;
                    
                    template<typename U>    
                    using is_rvalue = std::is_rvalue_reference<U>;

                    template<typename Msg>
                    using rvalue_func =  typename std::enable_if<is_rvalue<Msg>::value, request_cb_type>::type;

                    template<typename Msg>
                    using lvalue_func =  typename std::enable_if<!is_rvalue<Msg>::value, request_cb_type>::type;
                    
                    template<typename Message, typename CallBack>
                    lvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        using V = typename std::remove_reference_t<Message>::value_type;
                        return send(message_type{ref_message<V>{msg.data(),msg.size()}}, dst, tag, std::forward<CallBack>(callback));
                    }

                    template<typename Message, typename CallBack>
                    rvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback, std::true_type)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return send(message_type{std::move(msg)}, dst, tag, std::forward<CallBack>(callback));
                    }
	    
                    inline static void send_callback(void * __restrict ucx_req, ucs_status_t __restrict status)
                    {
                        auto& req = request_cb_data_type::get(ucx_req);
                        if (status == UCS_OK)
                            // call the callback
                            req.m_cb(std::move(req.m_msg), req.m_rank, req.m_tag);
                        // else: cancelled - do nothing
                        // set completion bit
                        //req.m_completed->m_ready = true;
                        *req.m_completed = true;
                        req.m_kind = request_kind::none;
                        // destroy the request - releases the message
                        req.~request_cb_data_type();
                        // free ucx request
                        //request_init(ucx_req);
				        ucp_request_free(ucx_req);
                    }

                    template<typename CallBack>
                    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        const auto& ep = m_send_worker->connect(dst);
                        const auto stag = ((std::uint_fast64_t)tag << 32) | 
                                           (std::uint_fast64_t)(rank());
                        auto ret = ucp_tag_send_nb(
                            ep.get(),                                        // destination
                            msg.data(),                                      // buffer
                            msg.size(),                                      // buffer size
                            ucp_dt_make_contig(1),                           // data type
                            stag,                                            // tag
                            &communicator::send_callback);                   // callback function pointer
                        
                        if (reinterpret_cast<std::uintptr_t>(ret) == UCS_OK)
                        {
                            // send operation is completed immediately and the call-back function is not invoked
                            // call the callback
                            callback(std::move(msg), dst, tag);
                            //return {nullptr, std::make_shared<request_cb_state_type>(true)};
                            return {};
                        } 
#ifndef GHEX_NO_EXCEPTIONS
                        else if(!UCS_PTR_IS_ERR(ret))
#endif
                        {
                            auto req_ptr = request_cb_data_type::construct(ret,
                                m_send_worker,
                                request_kind::send,
                                std::move(msg),
                                dst,
                                tag,
                                std::forward<CallBack>(callback),
                                std::make_shared<request_cb_state_type>(false));
                            return {req_ptr, req_ptr->m_completed};
                            /*auto& my_req = request_cb_data_type::get(ret);
                            my_req.m_ucx_ptr = ret;
                            my_req.m_worker = m_send_worker;
                            my_req.m_kind = request_kind::send;
                            my_req.m_msg = std::move(msg);
                            my_req.m_rank = dst;
                            my_req.m_tag = tag;
                            my_req.m_cb = std::forward<CallBack>(callback);
                            my_req.m_completed = std::make_shared<request_cb_state_type>(false);
                            return {&my_req, my_req.m_completed};*/
                        }
#ifndef GHEX_NO_EXCEPTIONS
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - send operation failed");
                        }
#endif
                    }
                    
                    template<typename Message, typename CallBack>
                    lvalue_func<Message&&> recv(Message&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        using V = typename std::remove_reference_t<Message>::value_type;
                        return recv(message_type{ref_message<V>{msg.data(),msg.size()}}, src, tag, std::forward<CallBack>(callback));
                    }

                    template<typename Message, typename CallBack>
                    rvalue_func<Message&&> recv(Message&& msg, rank_type src, tag_type tag, CallBack&& callback, std::true_type)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return recv(message_type{std::move(msg)}, src, tag, std::forward<CallBack>(callback));
                    }
	    
                    inline static void recv_callback(void * __restrict ucx_req, ucs_status_t __restrict status, ucp_tag_recv_info_t* /*info*/)
                    {
                        //const rank_type src = (rank_type)(info->sender_tag & 0x00000000fffffffful);
                        //const tag_type  tag = (tag_type)((info->sender_tag & 0xffffffff00000000ul) >> 32);
                        
                        auto& req = request_cb_data_type::get(ucx_req);

                        if (status == UCS_OK)
                        {
                            if (static_cast<int>(req.m_kind) == 0)
                            {
                                // we're in early completion mode
                                return;
                            }

                            req.m_cb(std::move(req.m_msg), req.m_rank, req.m_tag);
                            // set completion bit
                            //req.m_completed->m_ready = true;
                            *req.m_completed = true;
                            // destroy the request - releases the message
                            req.m_kind = request_kind::none;
                            req.~request_cb_data_type();
                            // free ucx request
                            //request_init(ucx_req);
                            ucp_request_free(ucx_req);
                        }
#ifndef GHEX_NO_EXCEPTIONS
                        else if (status == UCS_ERR_CANCELED)
#else
                        else
#endif
                        {
			                // canceled - do nothing
                            // set completion bit
                            //req.m_completed->m_ready = true;
                            *req.m_completed = true;
                            req.m_kind = request_kind::none;
                            // destroy the request - releases the message
                            req.~request_cb_data_type(); 
                            // free ucx request
                            //request_init(ucx_req);
                            ucp_request_free(ucx_req);
                        }
#ifndef GHEX_NO_EXCEPTIONS
                        else
                        {
                            // an error occurred
                            throw std::runtime_error("ghex: ucx error - recv message truncated");
                        }
#endif
                    }
                    
                    template<typename CallBack>
                    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        const auto rtag = ((std::uint_fast64_t)tag << 32) | 
                                           (std::uint_fast64_t)(src);
                        //using tp_t=std::remove_reference_t<decltype(m_send_worker->m_parallel_context->thread_primitives())>;
                        //using lk_t=typename tp_t::lock_type;
                        //lk_t lk(m_send_worker->m_parallel_context->thread_primitives().m_mutex);
                        return m_send_worker->m_parallel_context->thread_primitives().critical(
                            [this,rtag,&msg,src,tag,&callback]()
                            {
                                auto ret = ucp_tag_recv_nb(
                                    //m_recv_worker->get(),                            // worker
                                    m_ucp_rw,                                        // worker
                                    msg.data(),                                      // buffer
                                    msg.size(),                                      // buffer size
                                    ucp_dt_make_contig(1),                           // data type
                                    rtag,                                            // tag
                                    ~std::uint_fast64_t(0ul),                        // tag mask
                                    &communicator::recv_callback);                   // callback function pointer

#ifndef GHEX_NO_EXCEPTIONS
                                if(!UCS_PTR_IS_ERR(ret))
#endif
                                {
			                        if (UCS_INPROGRESS != ucp_request_check_status(ret))
                                    {
                                        // early completed
                                        callback(std::move(msg), src, tag);
		    		                    // we need to free the request here, not in the callback
                                        auto ucx_ptr = ret;
                                        //request_init(ucx_ptr);
                                        request_cb_data_type::get(ucx_ptr).m_kind = request_kind::none;
				                        ucp_request_free(ucx_ptr);
                                        //return request_cb_type{nullptr, std::make_shared<request_cb_state_type>(true)};
                                        return request_cb_type{};
                                    }
                                    else
                                    {
                                        auto req_ptr = request_cb_data_type::construct(ret,
                                            m_recv_worker,
                                            request_kind::recv,
                                            std::move(msg),
                                            src,
                                            tag,
                                            std::forward<CallBack>(callback),
                                            std::make_shared<request_cb_state_type>(false));
                                        return request_cb_type{req_ptr, req_ptr->m_completed};
                                        /*auto& my_req = request_cb_data_type::get(ret);
                                        my_req.m_ucx_ptr = ret;
                                        my_req.m_worker = m_recv_worker;
                                        my_req.m_kind = request_kind::recv;
                                        my_req.m_msg = std::move(msg);
                                        my_req.m_rank = src;
                                        my_req.m_tag = tag;
                                        my_req.m_cb = std::forward<CallBack>(callback);
                                        my_req.m_completed = std::make_shared<request_cb_state_type>(false);
                                        return request_cb_type{&my_req, my_req.m_completed};*/
                                    }
                                }
#ifndef GHEX_NO_EXCEPTIONS
                                else
                                {
                                    // an error occurred
                                    throw std::runtime_error("ghex: ucx error - recv operation failed");
                                }
#endif
                            }
                        );
                    }
                };
            } // namespace ucx

            template<typename ThreadPrimitives>
            struct transport_context<ucx_tag, ThreadPrimitives>
            {
            public: // member types
                using rank_type         = ucx::endpoint_t::rank_type;
                using worker_type       = ucx::worker_t<ThreadPrimitives>;
                using communicator_type = ucx::communicator<ThreadPrimitives>;

            private: // member types

                struct type_erased_address_db_t
                {
                    struct iface
                    {
                        virtual rank_type rank() = 0;
                        virtual rank_type size() = 0;
                        virtual int est_size() = 0;
                        virtual void init(const ucx::address_t&) = 0;
                        virtual ucx::address_t* find(rank_type) = 0;
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
                        ucx::address_t* find(rank_type rank) override { return m_impl.find(rank); }
                    };

                    std::unique_ptr<iface> m_impl;

                    template<typename Impl>
                    type_erased_address_db_t(Impl&& impl)
                    : m_impl{std::make_unique<impl_t<std::remove_cv_t<std::remove_reference_t<Impl>>>>(std::forward<Impl>(impl))}{}

                    inline rank_type rank() const { return m_impl->rank(); }
                    inline rank_type size() const { return m_impl->size(); }
                    inline int est_size() const { return m_impl->est_size(); }
                    inline void init(const ucx::address_t& addr) { m_impl->init(addr); }
                    inline ucx::address_t* find(rank_type rank) { return m_impl->find(rank); }
                };

                struct ucp_context_h_holder
                {
                    ucp_context_h m_context;
                    ~ucp_context_h_holder() { ucp_cleanup(m_context); }
                };

                using parallel_context_type = parallel_context<ThreadPrimitives>;
                using thread_token          = typename parallel_context_type::thread_token;
                using worker_vector         = std::vector<std::unique_ptr<worker_type>>;
                
            private: // members

                parallel_context_type&     m_parallel_context;
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
                transport_context(parallel_context<ThreadPrimitives>& pc, MPI_Comm, DB&& db, Args&&...)
                : m_parallel_context(pc)
                , m_db{std::forward<DB>(db)}
                , m_workers(m_parallel_context.thread_primitives().size())
                , m_tokens(m_parallel_context.thread_primitives().size())
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
                    m_worker = worker_type(this, &m_parallel_context, nullptr, UCS_THREAD_MODE_SERIALIZED);
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
                        m_workers[t.id()] = std::make_unique<worker_type>(this, &m_parallel_context, &m_tokens[t.id()], UCS_THREAD_MODE_SINGLE);
                    }
                    return {&m_worker, m_workers[t.id()].get()};
                }
                    
                rank_type rank() const { return m_db.rank(); }
                rank_type size() const { return m_db.size(); }
                ucp_context_h get() const noexcept { return m_context.m_context; }

            };

            //using mpi_context = context<mpi_tag>;

            namespace ucx {
                
                template<typename ThreadPrimitives>
                worker_t<ThreadPrimitives>::worker_t(transport_context_type* c, parallel_context_type* pc, thread_token* t, ucs_thread_mode_t mode)
                : m_context{c}
                , m_parallel_context{pc}
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
                }
                
                template<typename ThreadPrimitives>
                const endpoint_t& worker_t<ThreadPrimitives>::connect(rank_type rank)
                {
                    auto it = m_endpoint_cache.find(rank);
                    if (it != m_endpoint_cache.end())
                        return it->second;
#ifndef GHEX_NO_EXCEPTIONS
                    if (auto addr_ptr = m_context->m_db.find(rank))
#else

                    auto addr_ptr = m_context->m_db.find(rank);
#endif
                    {
                        auto p = m_endpoint_cache.insert(std::make_pair(rank, endpoint_t{rank, m_worker.get(), *addr_ptr}));
                        return p.first->second;
                    }
#ifndef GHEX_NO_EXCEPTIONS
                    else
                        throw std::runtime_error("could not connect to endpoint");
#endif
                }
            }

        }
    }
}

#endif /* INCLUDED_TL_MPI_CONTEXT_HPP */

