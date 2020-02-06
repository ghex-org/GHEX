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
#ifndef INCLUDED_GHEX_TL_COMMUNICATOR_HPP
#define INCLUDED_GHEX_TL_COMMUNICATOR_HPP

#include "callback_utils.hpp"
#include "message_buffer.hpp"
#include "shared_message_buffer.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            /** @brief This class combines the minimal backend-specific communicator implementation with the common
              * functionality for a low-level communicator.
              * @tparam Communicator backend communicator class*/
            template<class Communicator>
            class communicator : public Communicator
            {
            public: // member types
                using rank_type    = typename Communicator::rank_type;
                using tag_type     = typename Communicator::tag_type;
                using address_type = typename Communicator::address_type;
                using message_type = typename Communicator::message_type;
                template<typename T>
                using future       = typename Communicator::template future<T>;
                using request_cb   = typename Communicator::request_cb_type;
                template<typename V>
                using ref_message  = ::gridtools::ghex::tl::cb::ref_message<V>;
                template<typename U>    
                using is_rvalue    = std::is_rvalue_reference<U>;
                template<typename Msg, typename Ret = request_cb>
                using rvalue_func  =  typename std::enable_if<is_rvalue<Msg>::value, Ret>::type;
                template<typename Msg, typename Ret = request_cb>
                using lvalue_func  =  typename std::enable_if<!is_rvalue<Msg>::value, Ret>::type;

            public: // ctors
                template<typename...Args>
                communicator(Args&& ...args) : Communicator(std::forward<Args>(args)...) {}
                communicator(const communicator&) = default;
                communicator(communicator&&) = default;
                communicator& operator=(const communicator&) = default;
                communicator& operator=(communicator&&) = default;

            public: // member functions
                /** @brief generate a message buffer with a fixed size
                  * @tparam Allocator allocator type
                  * @param bytes size of buffer in bytes
                  * @param alloc allocator instance
                  * @return type erased message buffer */
                template<typename Allocator>
                static message_type make_message(std::size_t bytes, Allocator alloc) {
                    return message_buffer<Allocator>(bytes, alloc);
                }
                
                /** @brief generate a message buffer with a fixed size from a default constructed allocator
                  * @tparam Allocator allocator type
                  * @param bytes size of buffer in bytes
                  * @return type erased message buffer */
                template<typename Allocator = std::allocator<unsigned char>>
                static message_type make_message(std::size_t bytes) {
                    return make_message(bytes, Allocator{});
                }

                // use member functions from backend-specific base class
                using Communicator::send;
                using Communicator::recv;
                
                /** @brief send a shared message (shared pointer to a message) and get notified with a callback when the
                  * communication has finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a message type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param shared_msg_ptr a shared pointer to a message
                  * @param dst the destination rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                request_cb send(std::shared_ptr<Message>& shared_msg_ptr, rank_type dst, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return send(message_type{shared_msg_ptr}, dst, tag, std::forward<CallBack>(callback));
                }

                /** @brief send a shared message (shared_message_buffer) and get notified with a callback when the
                  * communication has finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Alloc an allocator type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param shared_msg a sheared message
                  * @param dst the destination rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Alloc, typename CallBack>
                request_cb send(shared_message_buffer<Alloc>& shared_msg, rank_type dst, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return send(message_type{shared_msg.m_message}, dst, tag, std::forward<CallBack>(callback));
                }
                
                /** @brief send a message and get notified with a callback when the communication has finished.
                  * The message must be kept alive by the caller until the communication is finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a meassage type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an l-value reference to the message to be sent
                  * @param dst the destination rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                lvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    using V = typename std::remove_reference_t<Message>::value_type;
                    return send(message_type{ref_message<V>{msg.data(),msg.size()}}, dst, tag, std::forward<CallBack>(callback));
                }

                /** @brief send a message and get notified with a callback when the communication has finished.
                  * The ownership of the message is transferred to this communicator and it is safe to destroy the
                  * message at the caller's site. 
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a meassage type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an r-value reference to the message to be sent
                  * @param dst the destination rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                rvalue_func<Message&&> send(Message&& msg, rank_type dst, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return send(message_type{std::move(msg)}, dst, tag, std::forward<CallBack>(callback));
                }

                /** @brief send a message to multiple destinations. The message must be kept alive by the caller until
                  * the communication is finished. 
                  * @tparam Message a message type
                  * @tparam Neighs a container class holding rank types
                  * @param msg an l-value reference to the message to be sent
                  * @param neighs a conainer of receivers
                  * @param tag the communication tag
                  * @return a vector of futures to test/wait for completion */
                template <typename Message, typename Neighs>
                [[nodiscard]] std::vector<future<void>> send_multi(Message& msg, const Neighs& neighs, tag_type tag) {
                    std::vector<future<void>> res;
                    res.reserve(neighs.size());
                    for (auto id : neighs)
                        res.push_back( send(msg, id, tag) );
                    return res;
                }

                /** @brief send a message to multiple destinations and get notified with a callback when the
                  * communication has finished. The message must be kept alive by the caller until the communication is
                  * finished. 
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a message type
                  * @tparam Neighs a container class holding rank types
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an l-value reference to the message to be sent
                  * @param neighs a conainer of receivers
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a vector of requests to thest (but not wait) for completion on each send operation */
                template <typename Message, typename Neighs, typename CallBack>
                lvalue_func<Message&&, std::vector<request_cb>>
                send_multi(Message&& msg, Neighs const &neighs, tag_type tag, const CallBack& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    std::vector<request_cb> res;
                    res.reserve(neighs.size());
                    auto counter = new std::atomic<int>(neighs.size());
                    for (auto id : neighs) {
                        res.push_back( send(msg, id, tag, 
                            [callback,counter](message_type m, rank_type r, tag_type t) {
                                if ( (--(*counter)) == 0) {
                                    callback(std::move(m),r,t);
                                    delete counter;
                                }
                            }) );
                    }
                    return res;
                }
                
                /** @brief send a message to multiple destinations and get notified with a callback when the
                  * communication has finished. The ownership of the message is transferred to this communicator and it
                  * is safe to destroy the message at the caller's site. 
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a message type
                  * @tparam Neighs a container class holding rank types
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an r-value reference to the message to be sent
                  * @param neighs a conainer of receivers
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a vector of requests to thest (but not wait) for completion on each send operation */
                template <typename Message, typename Neighs, typename CallBack>
                rvalue_func<Message&&, std::vector<request_cb>>
                send_multi(Message&& msg, Neighs const &neighs, tag_type tag, const CallBack& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    std::vector<request_cb> res;
                    res.reserve(neighs.size());
                    // keep message alive by making it shared
                    auto shared_msg = std::make_shared<Message>(std::move(msg));
                    auto counter = new std::atomic<int>(neighs.size());
                    for (auto id : neighs) {
                        res.push_back( send(shared_msg, id, tag, 
                            [callback, counter](message_type m, rank_type r, tag_type t) {
                                if ( (--(*counter)) == 0) {
                                    callback(std::move(m),r,t);
                                    delete counter;
                                }
                            }) );
                    }
                    return res;
                }

                /** @brief receive a shared message (shared pointer to a message) and get notified with a callback when the
                  * communication has finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a message type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param shared_msg_ptr a shared pointer to a message
                  * @param src the source rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                request_cb recv(std::shared_ptr<Message>& shared_msg_ptr, rank_type src, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return recv(message_type{shared_msg_ptr}, src, tag, std::forward<CallBack>(callback));
                }
                
                /** @brief receive a shared message (shared_message_buffer) and get notified with a callback when the
                  * communication has finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Alloc an allocator type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param shared_msg a sheared message
                  * @param src the source rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Alloc, typename CallBack>
                request_cb recv(shared_message_buffer<Alloc>& shared_msg, rank_type src, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return recv(message_type{shared_msg.m_message}, src, tag, std::forward<CallBack>(callback));
                }

                /** @brief receive a message and get notified with a callback when the communication has finished.
                  * The message must be kept alive by the caller until the communication is finished.
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a meassage type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an l-value reference to the message to be sent
                  * @param src the source rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                lvalue_func<Message&&> recv(Message&& msg, rank_type src, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    using V = typename std::remove_reference_t<Message>::value_type;
                    return recv(message_type{ref_message<V>{msg.data(),msg.size()}}, src, tag, std::forward<CallBack>(callback));
                }

                /** @brief receive a message and get notified with a callback when the communication has finished.
                  * The ownership of the message is transferred to this communicator and it is safe to destroy the
                  * message at the caller's site. 
                  * Note, that the communicator has to be progressed explicitely in order to guarantee completion.
                  * @tparam Message a meassage type
                  * @tparam CallBack a callback type with the signature void(message_type, rank_type, tag_type)
                  * @param msg an r-value reference to the message to be sent
                  * @param src the source rank
                  * @param tag the communication tag
                  * @param callback a callback instance
                  * @return a request to test (but not wait) for completion */
                template<typename Message, typename CallBack>
                rvalue_func<Message&&> recv(Message&& msg, rank_type src, tag_type tag, CallBack&& callback) {
                    GHEX_CHECK_CALLBACK_F(message_type,rank_type,tag_type) 
                    return recv(message_type{std::move(msg)}, src, tag, std::forward<CallBack>(callback));
                }
            };
        
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_COMMUNICATOR_HPP */

