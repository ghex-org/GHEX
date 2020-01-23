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
#ifndef INCLUDED_GHEX_TL_MPI_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_MPI_COMMUNICATOR_HPP

#include <boost/optional.hpp>
#include "../shared_message_buffer.hpp"
#include "../communicator.hpp"
#include "./future.hpp"
#include "./request_cb.hpp"
#include "../context.hpp"
#include "./communicator_state.hpp"

namespace gridtools {
    
    namespace ghex {

        namespace tl {
            
            template<typename ThreadPrimitives>
            struct transport_context<mpi_tag, ThreadPrimitives>;

            namespace mpi {

                template<typename ThreadPrimitives>
                class communicator {
                  public: // member types
                    using thread_primitives_type = ThreadPrimitives;
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using transport_context_type = typename shared_state_type::transport_context_type;
                    using thread_token = typename thread_primitives_type::token;
                    using state_type = communicator_state<ThreadPrimitives>;
                    using rank_type = typename state_type::rank_type;
                    using tag_type = typename state_type::tag_type;
                    using request = request_t;
                    using status = status_t;
                    template<typename T>
                    using future = typename state_type::template future<T>;
                    using address_type = rank_type;
                    using request_cb_type        = request_cb<ThreadPrimitives>;
                    using message_type           = typename request_cb_type::message_type;

                  private: // members
                    shared_state_type* m_shared_state;
                    state_type* m_state;

                  public: // ctors
                    communicator(shared_state_type* shared_state, state_type* state)
                    : m_shared_state{shared_state}
                    , m_state{state}
                    {}
                    communicator(const communicator&) = default;
                    communicator(communicator&&) = default;
                    communicator& operator=(const communicator&) = default;
                    communicator& operator=(communicator&&) = default;

                  public: // member functions
                    rank_type rank() const noexcept { return m_shared_state->rank(); }
                    rank_type size() const noexcept { return m_shared_state->size(); }
                    address_type address() const noexcept { return rank(); }

                    template<typename Message>
                    [[nodiscard]] future<void> send(const Message& msg, rank_type dst, tag_type tag) const {
                        request req;
                        GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(msg.data()),
                                                        sizeof(typename Message::value_type) * msg.size(), MPI_BYTE,
                                                        dst, tag, m_shared_state->m_comm, &req.get()));
                        return req;
                    }

                    template<typename Message>
                    [[nodiscard]] future<void> recv(Message& msg, rank_type src, tag_type tag) const {
                        request req;
                        GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(msg.data()),
                                                        sizeof(typename Message::value_type) * msg.size(), MPI_BYTE,
                                                        src, tag, m_shared_state->m_comm, &req.get()));
                        return req;
                    }

                    template <typename MsgType, typename Neighs>
                    std::vector<future<void>> send_multi(MsgType& msg, Neighs const &neighs, tag_type tag) const {
                        std::vector<future<void>> res;
                        res.reserve(neighs.size());
                        for (auto id : neighs)
                            res.push_back( send(msg, id, tag) );
                        return res;
                    }

                    unsigned progress() { return m_state->progress(); }

                    void barrier() {
                        if (auto token_ptr = m_state->m_token_ptr) {
                            auto& tp = *(m_shared_state->m_thread_primitives);
                            auto& token = *token_ptr;
                            tp.barrier(token);
                            tp.single(token, [this]() { MPI_Barrier(m_shared_state->m_comm); } );
                            tp.barrier(token);
                        }
                        else
                            MPI_Barrier(m_shared_state->m_comm);
                    }

                    template<typename V>
                    using ref_message = ::gridtools::ghex::tl::cb::ref_message<V>;
                    
                    template<typename U>    
                    using is_rvalue = std::is_rvalue_reference<U>;

                    template<typename Msg, typename Ret = request_cb_type>
                    using rvalue_func =  typename std::enable_if<is_rvalue<Msg>::value, Ret>::type;

                    template<typename Msg, typename Ret = request_cb_type>
                    using lvalue_func =  typename std::enable_if<!is_rvalue<Msg>::value, Ret>::type;

                    template<typename Message, typename CallBack>
                    request_cb_type send(std::shared_ptr<Message>& shared_msg_ptr, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return send(message_type{shared_msg_ptr}, dst, tag, std::forward<CallBack>(callback));
                    }

                    template<typename Alloc, typename CallBack>
                    request_cb_type send(shared_message_buffer<Alloc>& shared_msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return send(message_type{shared_msg.m_message}, dst, tag, std::forward<CallBack>(callback));
                    }
                    
                    template<typename Message, typename CallBack>
                    lvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        using V = typename std::remove_reference_t<Message>::value_type;
                        return send(message_type{ref_message<V>{msg.data(),msg.size()}}, dst, tag, std::forward<CallBack>(callback));
                    }

                    template<typename Message, typename CallBack>
                    rvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return send(message_type{std::move(msg)}, dst, tag, std::forward<CallBack>(callback));
                    }

                    template<typename CallBack>
                    request_cb_type send(message_type&& msg, rank_type dst, tag_type tag, CallBack&& callback)
                    {
                        auto fut = send(msg, dst, tag);
                        if (fut.ready())
                        {
                            callback(std::move(msg), dst, tag);
                            return {};
                        }
                        else
                        {
                            return { &m_state->m_send_queue,
                                m_state->m_send_queue.enqueue(std::move(msg), dst, tag, std::move(fut), 
                                        std::forward<CallBack>(callback))};
                        }
                    }
                    
                    template <typename Message, typename Neighs, typename CallBack>
                    lvalue_func<Message&&, std::vector<request_cb_type>>
                    send_multi(Message&& msg, Neighs const &neighs, tag_type tag, CallBack&& callback) {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        std::vector<request_cb_type> res;
                        res.reserve(neighs.size());
                        // keep callback alive by making it shared
                        using cb_type = typename std::remove_cv<typename std::remove_reference<CallBack>::type>::type;
                        auto cb_ptr = std::make_shared<cb_type>( std::forward<CallBack>(callback) );
                        for (auto id : neighs) {
                            res.push_back( send(std::forward<Message>(msg), id, tag, 
                                [cb = cb_ptr](message_type m, rank_type r, tag_type t) {
                                    if (cb.use_count() == 1)
                                        (*cb)(std::move(m),r,t);
                                }) );
                        }
                        return res;
                    }
                    
                    template <typename Message, typename Neighs, typename CallBack>
                    rvalue_func<Message&&, std::vector<request_cb_type>>
                    send_multi(Message&& msg, Neighs const &neighs, tag_type tag, CallBack&& callback) {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        std::vector<request_cb_type> res;
                        res.reserve(neighs.size());
                        // keep callback alive by making it shared
                        using cb_type = typename std::remove_cv<typename std::remove_reference<CallBack>::type>::type;
                        auto cb_ptr = std::make_shared<cb_type>( std::forward<CallBack>(callback) );
                        // keep message alive by making it shared
                        auto shared_msg = std::make_shared<Message>(std::move(msg));
                        for (auto id : neighs) {
                            res.push_back( send(shared_msg, id, tag, 
                                [cb = cb_ptr](message_type m, rank_type r, tag_type t) {
                                    if (cb.use_count() == 1)
                                        (*cb)(std::move(m),r,t);
                                }) );
                        }
                        return res;
                    }

                    template<typename Message, typename CallBack>
                    request_cb_type recv(std::shared_ptr<Message>& shared_msg_ptr, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return recv(message_type{shared_msg_ptr}, src, tag, std::forward<CallBack>(callback));
                    }
                    
                    template<typename Alloc, typename CallBack>
                    request_cb_type recv(shared_message_buffer<Alloc>& shared_msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                        return recv(message_type{shared_msg.m_message}, src, tag, std::forward<CallBack>(callback));
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

                    template<typename CallBack>
                    request_cb_type recv(message_type&& msg, rank_type src, tag_type tag, CallBack&& callback)
                    {
                        auto fut = recv(msg, src, tag);
                        if (fut.ready())
                        {
                            callback(std::move(msg), src, tag);
                            return {};
                        }
                        else
                        {
                            return { &m_state->m_recv_queue,
                                m_state->m_recv_queue.enqueue(std::move(msg), src, tag, std::move(fut), 
                                        std::forward<CallBack>(callback))};
                        }
                    }
                };

            } // namespace mpi

        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_HPP */

