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
#ifndef INCLUDED_CALLBACK_COMMUNICATOR_HPP
#define INCLUDED_CALLBACK_COMMUNICATOR_HPP

#include <deque>
#include <algorithm>
#include <memory>
#include <type_traits>
#include <tuple>
#include "message.hpp"


namespace gridtools
{
    namespace ghex
    {

        /** callback_communicator is a class to dispatch send and receive operations to. Each operation can optionally 
          * be tied to a user defined callback function / function object. The payload of each send/receive operation
          * must be a ghex::shared_message<Allocator>. This class will keep a (shallow) copy of each message, thus
          * it is safe to release the message at the caller's site.
          *
          * The communication must be explicitely progressed using the function call operator().
          *
          * An instance of this class is 
          * - a move-only.
          * - not thread-safe
          *
          * If unprogressed operations remain at time of destruction, std::terminate will be called.
          *
          * @tparam Communicator underlying transport communicator
          * @tparam Allocator    allocator type used for allocating shared messages */
        template<class Communicator, class Allocator = std::allocator<unsigned char>>
        class callback_communicator
        {
        public: // member types
            
            using communicator_type = Communicator;
            using future_type       = typename communicator_type::future_type;
            using tag_type          = typename communicator_type::tag_type;
            using rank_type         = typename communicator_type::rank_type;
            using allocator_type    = Allocator;
            using message_type      = mpi::raw_shared_message<allocator_type>;

        private: // member types

            // necessary meta information for each send/receive operation
            struct element_type
            {
                using message_arg_type = const message_type&;
                std::function<void(rank_type, tag_type, message_arg_type)> m_cb;
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

            communicator_type   &m_comm;
            allocator_type      m_alloc;
            send_container_type m_sends;
            recv_container_type m_recvs; 

        public: // ctors

            callback_communicator(communicator_type& comm, allocator_type alloc = allocator_type{}) 
                : m_comm(comm), m_alloc(alloc) {}

            // callback_communicator(communicator_type&& comm, allocator_type alloc = allocator_type{}) 
            //     : m_comm(std::move(comm)), m_alloc(alloc) {}
            callback_communicator(const callback_communicator&) = delete;
            callback_communicator(callback_communicator&&) = default;
            ~callback_communicator() 
            { 
                if (m_sends.size() != 0 || m_recvs.size() != 0)  
                {
                    // std::terminate(); 
                }
            }
            
        public: // queries

            std::size_t pending_sends() const { return m_sends.size(); }
            std::size_t pending_recvs() const { return m_recvs.size(); }

        public: // send

            /** @brief Send a message to a destination with the given tag and register a callback which will be invoked
              * when the send operation is completed.
              * @tparam CallBack User defined callback class which defines void Callback::operator()(rank_type,tag_type,const message_type&)
              * @param msg Message to be sent
              * @param dst Destination of the message
              * @param tag Tag associated with the message
              * @param cb  Callback function object */
            template<typename CallBack>
            void send(const message_type& msg, rank_type dst, tag_type tag, CallBack&& cb)
            {
                m_sends.push_back( send_element_type{ std::forward<CallBack>(cb), dst, tag, m_comm.send(msg, dst, tag), msg } );
            }

            /** @brief Send a message without registering a callback. */
            void send(const message_type& msg, rank_type dst, tag_type tag)
            {
                send(msg,dst,tag,[](rank_type,tag_type,const message_type&){});
            }

            /** @brief Send a message to multiple destinations with the same rank an register an associated callback. 
              * @tparam Neighs Range over rank_type
              * @tparam CallBack User defined callback class which defines void Callback::operator()(rank_type,tag_type,const message_type&)
              * @param msg Message to be sent
              * @param neighs Range of destination ranks
              * @param tag Tag associated with the message
              * @param cb Callback function object */
            template <typename Neighs, typename CallBack>
            void send_multi(const message_type& msg, Neighs const &neighs, int tag, CallBack&& cb)
            {
                std::shared_ptr<CallBack> cb_ptr(new CallBack{std::forward<CallBack>(cb)});
                for (auto id : neighs)
                    send(msg, id, tag, 
                            [cb_ptr](rank_type r, tag_type t, const message_type& m)
                            {
                                // if (cb_ptr->use_count == 1)
                                (*cb_ptr)(r,t,m); 
                            });
            }

            /** @brief Send a message to multiple destination without registering a callback */
            template <typename Neighs>
            void send_multi(const message_type& msg, Neighs const &neighs, int tag)
            {
                send_multi(msg,neighs,tag,[](rank_type,tag_type,const message_type&){});
            }

        public: // recieve

            /** @brief Receive a message from a source rank with the given tag and register a callback which will be invoked
              * when the receive operation is completed.
              * @tparam CallBack User defined callback class which defines void Callback::operator()(rank_type,tag_type,const message_type&)
              * @param msg Message where data will be received
              * @param src Source of the message
              * @param tag Tag associated with the message
              * @param cb  Callback function object */
            template<typename CallBack>
            void recv(const message_type& msg, rank_type src, tag_type tag, CallBack&& cb)
            {
                m_recvs.push_back( recv_element_type{ std::forward<CallBack>(cb), src, tag, m_comm.recv(msg, src, tag), msg } );
            }

            /** @brief Receive a message with length size (storage is allocated accordingly). */
            template<typename CallBack>
            void recv(std::size_t size, rank_type src, tag_type tag, CallBack&& cb)
            {
                recv(message_type{size,size}, src, tag, std::forward<CallBack>(cb));
            }

            /** @brief Receive a message without registering a callback. */
            void recv(const message_type& msg, rank_type src, tag_type tag)
            {
                recv(msg,src,tag,[](rank_type,tag_type,const message_type&){});
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
              * completed and calls the associated callback (if it exists). When all registered operations have been
              * completed this function checks for further unexpected incoming messages which will be received in a
              * newly allocated shared_message and returned to the user through invocation of the provided callback.
              * @tparam CallBack User defined callback class which defines void Callback::operator()(rank_type,tag_type,const message_type&)
              * @param unexpected_cb Callback function object
              * @return returns false if all registered operations have been completed. */
            template<typename CallBack>
            bool progress(CallBack&& unexpected_cb)
            {
                const auto not_completed = progress();
                if (!not_completed)
                {
                    if (auto o = m_comm.recv_any(m_alloc))
                    {
                        const auto r = std::get<0>(*o);
                        const auto t = std::get<1>(*o);
                        auto msg = std::get<2>(*o);
                        unexpected_cb(r,t,msg);
                    }
                }
                return not_completed;
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

            bool cancel_sends() { return cancel(m_sends); }
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
			// TODO: remove, not needed
                        // element.m_future.wait();
                        element.m_cb(element.m_rank, element.m_tag, element.m_msg);
                        break;
                    }
                    else
                    {
                        d.push_back(std::move(element));
                    }
                }
                return (d.size()==0u);
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

    } // namespace ghex
}// namespace gridtools

#endif /* INCLUDED_CALLBACK_COMMUNICATOR_HPP */

