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
#ifndef INCLUDED_GHEX_TL_CALLBACK_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_CALLBACK_COMMUNICATOR_HPP

#include <deque>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <tuple>
#include <boost/callable_traits.hpp>
#include <boost/optional.hpp>

#include "../common/debug.hpp"

#ifdef USE_RAW_SHARED_MESSAGE
#include "./raw_shared_message_buffer.hpp"
#else
#include "./shared_message_buffer.hpp"
#endif

/** @brief checks the arguments of callback function object */
#define GHEX_CHECK_CALLBACK                                                                   \
    using args_t = boost::callable_traits::args_t<CallBack>;                                  \
    using arg0_t = std::tuple_element_t<0, args_t>;                                           \
    using arg1_t = std::tuple_element_t<1, args_t>;                                           \
    using arg2_t = std::tuple_element_t<2, args_t>;                                           \
    static_assert(std::tuple_size<args_t>::value==3,                                          \
        "callback must have 3 arguments");                                                    \
    static_assert(std::is_convertible<arg1_t,rank_type>::value,                               \
        "rank_type is not convertible to second callback argument type");                     \
    static_assert(std::is_convertible<arg2_t,tag_type>::value,                                \
        "tag_type is not convertible to third callback argument type");                       \
    static_assert(std::is_convertible<arg0_t,typename element_type::message_arg_type>::value, \
        "first callback argument type is not a message_type");

namespace gridtools
{
    namespace ghex
    {
        namespace tl {

            /** callback_communicator is a class to dispatch send and receive operations to. Each operation can 
              * optionally be tied to a user defined callback function / function object. The payload of each 
              * send/receive operation must be a ghex::shared_message_buffer<Allocator>. 
              * This class will keep a (shallow) copy of each message, thus it is safe to release the message at 
              * the caller's site.
              *
              * The user defined callback must define void operator()(message_type,rank_type,tag_type), where
              * message_type is a shared_message_buffer that can be cheaply copied/moved from within the callback body 
              * if needed.
              *
              * The communication must be explicitely progressed using the member function progress.
              *
              * An instance of this class is 
              * - a move-only.
              * - not thread-safe
              *
              * @tparam Communicator underlying transport communicator
              * @tparam Allocator    allocator type used for allocating shared message buffers */
            template<class Communicator, class Allocator = std::allocator<unsigned char>>
            class callback_communicator
            {
            public: // member types
                
                using communicator_type = Communicator;
                using future_type       = typename communicator_type::template future<void>;
                using tag_type          = typename communicator_type::tag_type;
                using rank_type         = typename communicator_type::rank_type;
                using allocator_type    = Allocator;
                using message_type      = shared_message_buffer<allocator_type>;

            private: // member types

                // necessary meta information for each send/receive operation
                struct element_type
                {
                    using message_arg_type = message_type;
                    std::function<void(message_arg_type, rank_type, tag_type)> m_cb;
                    rank_type    m_rank;
                    tag_type     m_tag;
                    future_type  m_future;
                    message_type m_msg;
                };
                using send_element_type   = element_type;
                using recv_element_type   = element_type;
                using send_container_type = std::deque<send_element_type>;
                using recv_container_type = std::deque<recv_element_type>;

            private: // members

                communicator_type   m_comm;
                allocator_type      m_alloc;
                send_container_type m_sends;
                recv_container_type m_recvs; 

            public: // ctors

                /** @brief construct from a basic transport communicator
                  * @param comm  the underlying transport communicator
                  * @param alloc the allocator instance to be used for constructing messages */
                callback_communicator(const communicator_type& comm, allocator_type alloc = allocator_type{}) 
                : m_comm(comm), m_alloc(alloc) {}
                callback_communicator(communicator_type&& comm, allocator_type alloc = allocator_type{}) 
                : m_comm(std::move(comm)), m_alloc(alloc) {}

                callback_communicator(const callback_communicator&) = delete;
                callback_communicator(callback_communicator&&) = default;

                /** terminates the program if the queues are not empty */
                ~callback_communicator() 
                { 
                    if (m_sends.size() != 0 || m_recvs.size() != 0)  
                    {
                        WARN("Uncompleted communication requests still in the queue in ~callback_communicator()");
                    }
                }
                
            public: // queries

                /** returns the number of unprocessed send handles in the queue. */
                std::size_t pending_sends() const { return m_sends.size(); }
                /** returns the number of unprocessed recv handles in the queue. */
                std::size_t pending_recvs() const { return m_recvs.size(); }

            public: // get a message

                /** get a message with size n from the communicator */
                message_type make_message(std::size_t n = 0u) const
                {
                    return { m_alloc, n };
                }

            public: // send

                /** @brief Send a message to a destination with the given tag and register a callback which will be 
                  * invoked when the send operation is completed.
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(message_type,rank_type,tag_type)
                  * @param msg Message to be sent
                  * @param dst Destination of the message
                  * @param tag Tag associated with the message
                  * @param cb  Callback function object */
                template<typename CallBack>
                void send(message_type msg, rank_type dst, tag_type tag, CallBack&& cb)
                {
                    GHEX_CHECK_CALLBACK
                    auto& element = *m_sends.insert(m_sends.end(), send_element_type{std::forward<CallBack>(cb), dst, tag,
                                                                                     future_type{}, std::move(msg)});
                    element.m_future = std::move( m_comm.send(element.m_msg, dst, tag) );
                }

                /** @brief Send a message without registering a callback. */
                void send(message_type msg, rank_type dst, tag_type tag)
                {
                    send(std::move(msg),dst,tag,[](message_type,rank_type,tag_type){});
                }

                /** @brief Send a message to multiple destinations with the same rank an register an associated callback. 
                  * @tparam Neighs Range over rank_type
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(rank_type,tag_type,message_type)
                  * @param msg Message to be sent
                  * @param neighs Range of destination ranks
                  * @param tag Tag associated with the message
                  * @param cb Callback function object */
                template <typename Neighs, typename CallBack>
                void send_multi(message_type msg, Neighs const &neighs, int tag, CallBack&& cb)
                {
                    GHEX_CHECK_CALLBACK
                    using cb_type = typename std::remove_cv<typename std::remove_reference<CallBack>::type>::type;
                    auto cb_ptr = std::make_shared<cb_type>( std::forward<CallBack>(cb) );
                    for (auto id : neighs)
                        send(msg, id, tag,
                                [cb_ptr](message_type m, rank_type r, tag_type t)
                                {
                                    // if (cb_ptr->use_count == 1)
                                    (*cb_ptr)(std::move(m),r,t); 
                                });
                }

                /** @brief Send a message to multiple destinations without registering a callback */
                template <typename Neighs>
                void send_multi(message_type msg, Neighs const &neighs, int tag)
                {
                    send_multi(std::move(msg),neighs,tag,[](message_type, rank_type,tag_type){});
                }

            public: // recieve

                /** @brief Receive a message from a source rank with the given tag and register a callback which will
                  * be invoked when the receive operation is completed.
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(message_type,rank_type,tag_type)
                  * @param msg Message where data will be received
                  * @param src Source of the message
                  * @param tag Tag associated with the message
                  * @param cb  Callback function object */
                template<typename CallBack>
                void recv(message_type msg, rank_type src, tag_type tag, CallBack&& cb)
                {
                    GHEX_CHECK_CALLBACK
                    auto& element = *m_recvs.insert(m_recvs.end(), recv_element_type{std::forward<CallBack>(cb), src, tag,
                                                                                     future_type{}, std::move(msg)});
                    element.m_future = std::move( m_comm.recv(element.m_msg, src, tag) );
                }

                /** @brief Receive a message with length size (storage is allocated accordingly). */
                template<typename CallBack>
                void recv(std::size_t size, rank_type src, tag_type tag, CallBack&& cb)
                {
                    recv(message_type{size,m_alloc}, src, tag, std::forward<CallBack>(cb));
                }

                /** @brief Receive a message without registering a callback. */
                void recv(message_type msg, rank_type src, tag_type tag)
                {
                    recv(std::move(msg),src,tag,[](message_type,rank_type,tag_type){});
                }

            public: // progress

                /** @brief Progress the communication. This function checks whether any receive and send operation is 
                  * completed and calls the associated callback (if it exists).
                  * @return returns false if all registered operations have been completed.*/
                bool progress()
                {
                    const auto sends_completed = run(m_sends);
                    const auto recvs_completed = run(m_recvs);
                    const auto completed = sends_completed && recvs_completed;
                    return !completed;
                }

                /** @brief Progress the communication. This function checks whether any receive and send operation is 
                  * completed and calls the associated callback (if it exists). When all registered operations have 
                  * been completed this function checks for further unexpected incoming messages which will be received 
                  * in a newly allocated shared_message_buffer and returned to the user through invocation of the 
                  * provided callback.
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(message_type,rank_type,tag_type)
                  * @param unexpected_cb callback function object
                  * @return returns false if all registered operations have been completed. */
                template<typename CallBack>
                bool progress(CallBack&& unexpected_cb)
                {
                    GHEX_CHECK_CALLBACK
                    const auto not_completed = progress();
                    if (!not_completed)
                    {
                        if (auto o = m_comm.template recv_any_source_any_tag<message_type>(m_alloc))
                        {
                            auto t = o->get();
                            unexpected_cb(std::move(std::get<2>(t)),std::get<0>(t),std::get<1>(t));
                        }
                    }
                    return not_completed;
                }

            public: // attach/detach
                
                /** @brief Deregister a send operation from this object which matches the given destination and tag.
                  * If such operation is found the callback will be discared and the message will be returned to the
                  * caller together with a future on which completion can be awaited.
                  * @param dst Destination of the message
                  * @param tag Tag associated with the message
                  * @return Either a pair of future and message or none */
                boost::optional<std::pair<future_type,message_type>> detach_send(rank_type dst, tag_type tag)
                {
                    return detach(dst,tag,m_sends);
                }

                /** @brief Deregister a receive operation from this object which matches the given destination and tag.
                  * If such operation is found the callback will be discared and the message will be returned to the 
                  * caller together with a future on which completion can be awaited.
                  * @param src Source of the message
                  * @param tag Tag associated with the message
                  * @return Either a pair of future and message or none */
                boost::optional<std::pair<future_type,message_type>> detach_recv(rank_type src, tag_type tag)
                {
                    return detach(src,tag,m_recvs);
                }

                /** @brief Register a send operation with this object with future, destination and tag and associate it
                  * with a callback. This is the inverse operation of detach. Note, that attaching of a send operation
                  * originating from the underlying basic communicator is supported.
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(message_type,rank_type,tag_type)
                  * @param fut future object
                  * @param msg message data
                  * @param dst destination rank
                  * @param tag associated tag
                  * @param cb  Callback function object */
                template<typename CallBack>
                void attach_send(future_type&& fut, message_type msg, rank_type dst, tag_type tag, CallBack&& cb)
                {
                    GHEX_CHECK_CALLBACK
                    m_sends.push_back( send_element_type{ std::forward<CallBack>(cb), dst, tag, std::move(fut), std::move(msg) } );
                }

                /** @brief Register a send without associated callback. */
                void attach_send(future_type&& fut, message_type msg, rank_type dst, tag_type tag)
                {
                    m_sends.push_back( send_element_type{ [](message_type,rank_type,tag_type){}, dst, tag, std::move(fut), std::move(msg) } );
                }

                /** @brief Register a receive operation with this object with future, source and tag and associate it
                  * with a callback. This is the inverse operation of detach. Note, that attaching of a recv operation
                  * originating from the underlying basic communicator is supported.
                  * @tparam CallBack User defined callback class which defines 
                  *                  void Callback::operator()(message_type,rank_type,tag_type)
                  * @param fut future object
                  * @param msg message data
                  * @param dst source rank
                  * @param tag associated tag
                  * @param cb  Callback function object */
                template<typename CallBack>
                void attach_recv(future_type&& fut, message_type msg, rank_type src, tag_type tag, CallBack&& cb)
                {
                    GHEX_CHECK_CALLBACK
                    m_recvs.push_back( send_element_type{ std::forward<CallBack>(cb), src, tag, std::move(fut), std::move(msg) } );
                }

                /** @brief Register a receive without associated callback. */
                void attach_recv(future_type&& fut, message_type msg, rank_type src, tag_type tag)
                {
                    m_recvs.push_back( send_element_type{ [](message_type,rank_type,tag_type){}, src, tag, std::move(fut), std::move(msg) } );
                }

            public: // cancel
                
                /** @brief Deregister all operations from this object and attempt to cancel the communication.
                  * @return true if cancelling was successful. */
                bool cancel()
                {
                    const auto s = cancel_sends();
                    const auto r = cancel_recvs();
                    return s && r;
                }

                /** @brief Deregister all send operations from this object and attempt to cancel the communication.
                  * @return true if cancelling was successful. */
                bool cancel_sends() { return cancel(m_sends); }

                /** @brief Deregister all recv operations from this object and attempt to cancel the communication.
                  * @return true if cancelling was successful. */
                bool cancel_recvs() { return cancel(m_recvs); }

            private: // implementation

                template<typename Deque>
                bool run(Deque& d)
                {
                    const unsigned int size = d.size();
                    for (unsigned int i=0; i<size; ++i) 
                    {
                        auto element = std::move(d.front());
                        d.pop_front();

                        if (element.m_future.ready())
                        {
                            //element.m_future.wait();
                            element.m_cb(std::move(element.m_msg), element.m_rank, element.m_tag);
                            //break;
                        }
                        else
                        {
                            d.push_back(std::move(element));
                        }
                    }
                    return (d.size()==0u);
                }

                template<typename Deque>
                boost::optional<std::pair<future_type,message_type>> detach(rank_type rank, tag_type tag, Deque& d)
                {
                    auto it = std::find_if(d.begin(), d.end(), 
                        [rank, tag](auto const& x) 
                        {
                            return (x.m_rank == rank && x.m_tag == tag);
                        });
                    if (it != d.end())
                    {
                        auto cb =  std::move(it->m_cb);
                        auto fut = std::move(it->m_future);
                        auto msg = std::move(it->m_msg);
                        d.erase(it);
                        return std::pair<future_type,message_type>{std::move(fut), std::move(msg)}; 
                    }
                    return boost::none;
                }

                template<typename Deque>
                bool cancel(Deque& d)
                {
                    bool result = true;
                    const unsigned int size = d.size();
                    for (unsigned int i=0; i<size; ++i) 
                    {
                        auto element = std::move(d.front());
                        d.pop_front();
                        auto& fut = element.m_future;
                        if (!fut.ready())
                            result = result && fut.cancel();
                        else
                            fut.wait();
                    }
                    return result;
                }
            };

        } // namespace tl
    } // namespace ghex
}// namespace gridtools

#endif /* INCLUDED_GHEX_TL_CALLBACK_COMMUNICATOR_HPP */

