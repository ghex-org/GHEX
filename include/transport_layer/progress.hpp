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
#ifndef INCLUDED_PROGRESS_HPP
#define INCLUDED_PROGRESS_HPP

#include <deque>
#include <memory>
#include <type_traits>
#include <tuple>
#include <boost/callable_traits.hpp>
#include <boost/optional.hpp>
#include <iostream>
#include "./mpi/message.hpp"

namespace gridtools
{
    namespace ghex
    {

        template<class Communicator, class Allocator = std::allocator<unsigned char>>
        class progress
        {
        public: // member types
            
            using communicator_type = Communicator;
            using future_type       = typename communicator_type::future_type;
            using tag_type          = typename communicator_type::tag_type;
            using rank_type         = typename communicator_type::rank_type;
            using allocator_type    = typename std::allocator_traits<Allocator>::template rebind_alloc<unsigned char>;
            using message_type      = mpi::shared_message<allocator_type>;

        private: // member types

            struct send_element_type
            {
                using message_arg_type = const message_type&;
                std::function<void(rank_type, tag_type, message_arg_type)> m_cb;
                rank_type    m_rank;
                tag_type     m_tag;
                future_type  m_future;
                message_type m_msg;
            };
            struct recv_element_type
            {
                using message_arg_type = const message_type&;
                std::function<void(rank_type, tag_type, message_arg_type)> m_cb;
                rank_type    m_rank;
                tag_type     m_tag;
                future_type  m_future;
                message_type m_msg;
            };
            using send_container_type = std::deque<send_element_type>;
            using recv_container_type = std::deque<recv_element_type>;

        private: // members

            communicator_type   m_comm;
            send_container_type m_sends;
            recv_container_type m_recvs; 

        public: // ctors
            progress(const communicator_type& comm) : m_comm(comm) {}
            progress(communicator_type&& comm) : m_comm(std::move(comm)) {}
            progress(const progress&) = delete;
            progress(progress&&) = default;
            ~progress() 
            { 
                if (m_sends.size() != 0 || m_recvs.size() != 0)  
                {
                    //std::cout << "not empty: " << m_sends.size() << ", " << m_recvs.size() << std::endl;
                    std::terminate(); 
                }
            }
            
        public: // member functions

            std::size_t size_sends() const { return m_sends.size(); }
            std::size_t size_recvs() const { return m_recvs.size(); }

            template<typename CallBack>
            void send(const message_type& msg, rank_type dst, tag_type tag, CallBack&& cb)
            {
                // static checks
                using args_t = boost::callable_traits::args_t<CallBack>;
                using arg0_t = std::tuple_element_t<0, args_t>;
                using arg1_t = std::tuple_element_t<1, args_t>;
                using arg2_t = std::tuple_element_t<2, args_t>;
                static_assert(std::tuple_size<args_t>::value==3, "callback must have 3 arguments");
                static_assert(std::is_convertible<arg0_t,rank_type>::value, "rank_type is not convertible to first callback argument type");
                static_assert(std::is_convertible<arg1_t,tag_type>::value, "tag_type is not convertible to second callback argument type");
                static_assert(std::is_same<arg2_t,typename send_element_type::message_arg_type>::value, "third callback argument type is not a const reference of message_type");
                // add to list
                m_sends.push_back( send_element_type{ std::forward<CallBack>(cb), dst, tag, m_comm.send(msg, dst, tag), msg } );
            }

            template <typename Neighs, typename CallBack>
            void send_multi(const message_type& msg, Neighs const &neighs, int tag, CallBack&& cb)
            {
                for (auto id : neighs)
                    send(msg, id, tag, std::forward<CallBack>(cb));
            }

            template <typename Neighs>
            void send_multi(const message_type& msg, Neighs const &neighs, int tag)
            {
                send_multi(msg,neighs,tag,[](rank_type,tag_type,const message_type&){});
            }

            template<typename CallBack>
            void recv(const message_type& msg, rank_type src, tag_type tag, CallBack&& cb)
            {
                // static checks
                using args_t = boost::callable_traits::args_t<CallBack>;
                using arg0_t = std::tuple_element_t<0, args_t>;
                using arg1_t = std::tuple_element_t<1, args_t>;
                using arg2_t = std::tuple_element_t<2, args_t>;
                static_assert(std::tuple_size<args_t>::value==3, "callback must have 3 arguments");
                static_assert(std::is_convertible<arg0_t,rank_type>::value, "rank_type is not convertible to first callback argument type");
                static_assert(std::is_convertible<arg1_t,tag_type>::value, "tag_type is not convertible to second callback argument type");
                static_assert(std::is_same<arg2_t,typename recv_element_type::message_arg_type>::value, "third callback argument type is not a const reference of message_type");
                // add to list
                m_recvs.push_back( recv_element_type{ std::forward<CallBack>(cb), src, tag, m_comm.recv(msg, src, tag), msg } );
            }

            bool operator()()
            {
                const auto sends_completed = run(m_sends);
                const auto recvs_completed = run(m_recvs);
                const auto completed = sends_completed && recvs_completed;
                return !completed;
            }
            
            boost::optional<std::pair<future_type,message_type>> detach_send(rank_type dst, tag_type tag)
            {
                return detach(dst,tag,m_sends);
            }

            boost::optional<std::pair<future_type,message_type>> detach_recv(rank_type dst, tag_type tag)
            {
                return detach(dst,tag,m_recvs);
            }
            
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
                        element.m_future.wait();
                        auto f = std::move( element.m_cb);
                        f(element.m_rank, element.m_tag, element.m_msg);
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
                    return std::pair<future_type,message_type>{std::move(fut), msg}; 
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

    } // namespace ghex
}// namespace gridtools

#endif /* INCLUDED_PROGRESS_HPP */

