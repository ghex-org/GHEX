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
//#include "./communicator_base.hpp"
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
                    using shared_state_type = shared_communicator_state<ThreadPrimitives>;
                    using transport_context_type = typename shared_state_type::transport_context_type;
                    using parallel_context_type = typename shared_state_type::parallel_context_type;
                    using thread_token = typename parallel_context_type::thread_token;
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

                    template <typename Neighs>
                    std::vector<future<void>> send_multi(const message_type& msg, Neighs const &neighs, int tag) const {
                        std::vector<future<void>> res;
                        res.reserve(neighs.size());
                        for (auto id : neighs)
                            res.push_back( send(msg, id, tag) );
                        return res;
                    }

                    unsigned progress() { return m_state->progress(); }

                    void barrier() {
                        if (auto token_ptr = m_state->m_token_ptr) {
                            auto& tp = m_shared_state->m_parallel_context->thread_primitives();
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

                    template<typename Msg>
                    using rvalue_func =  typename std::enable_if<is_rvalue<Msg>::value, request_cb_type>::type;

                    template<typename Msg>
                    using lvalue_func =  typename std::enable_if<!is_rvalue<Msg>::value, request_cb_type>::type;

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


                ///** Mpi communicator which exposes basic non-blocking transport functionality and 
                //  * returns futures to await said transports to complete. */
                //template<typename ThreadPrimitives>
                //class communicator//<mpi_tag,ThreadPrimitives>
                //: public communicator_base
                //{
                //public:
                //    using transport_type = mpi_tag;
                //    using base_type      = mpi::communicator_base;
                //    using address_type   = typename base_type::rank_type;
                //    using rank_type      = typename base_type::rank_type;
                //    using size_type      = typename base_type::size_type;
                //    using tag_type       = typename base_type::tag_type;
                //    using request        = request_t;
                //    using status         = status_t;
                //    template<typename T>
                //    using future         = future_t<T>;

                //public:

                //    using transport_context_type = transport_context<mpi_tag, ThreadPrimitives>;
                //    transport_context_type* m_transport_context;
                //    int m_thread_id;

                //    communicator(const MPI_Comm& c, transport_context_type* tc, int thread_id = -1) 
                //    : base_type{c} 
                //    , m_transport_context{tc}
                //    , m_thread_id{thread_id}
                //    {}
                //    
                //    communicator(const communicator&) = default;
                //    communicator(communicator&&) noexcept = default;

                //    communicator& operator=(const communicator&) = default;
                //    communicator& operator=(communicator&&) noexcept = default;

                //    /** @return address of this process */
                //    address_type address() const { return rank(); }

                //public: // send

                //    /** @brief non-blocking send
                //      * @tparam Message a container type
                //      * @param msg source container
                //      * @param dest destination rank
                //      * @param tag message tag
                //      * @return completion handle */
                //    template<typename Message> 
                //    [[nodiscard]] future<void> send(const Message& msg, rank_type dest, tag_type tag) const
                //    {
                //        request req;
                //        GHEX_CHECK_MPI_RESULT(
                //            MPI_Isend(reinterpret_cast<const void*>(msg.data()),sizeof(typename Message::value_type)*msg.size(), 
                //                      MPI_BYTE, dest, tag, *this, &req.get())
                //        );
                //        return req;
                //    }
                //
                //public: // recv

                //    /** @brief non-blocking receive
                //      * @tparam Message a container type
                //      * @param msg destination container
                //      * @param source source rank
                //      * @param tag message tag
                //      * @return completion handle */
                //    template<typename Message>
                //    [[nodiscard]] future<void> recv(Message& msg, rank_type source, tag_type tag) const
                //    {
                //        request req;
                //        GHEX_CHECK_MPI_RESULT(
                //                MPI_Irecv(reinterpret_cast<void*>(msg.data()),sizeof(typename Message::value_type)*msg.size(), 
                //                          MPI_BYTE, source, tag, *this, &req.get()));
                //        return req;
                //    }

                //    /** @brief non-blocking receive which allocates the container within this function and returns it
                //      * in the future 
                //      * @tparam Message a container type
                //      * @tparam Args additional argument types for construction of Message
                //      * @param n number of elements to be received
                //      * @param source source rank
                //      * @param tag message tag
                //      * @param args additional arguments to be passed to new container of type Message at construction 
                //      * @return completion handle with message as payload */
                //    template<typename Message, typename... Args>
                //    [[nodiscard]] future<Message> recv(int n, rank_type source, tag_type tag, Args&& ...args) const
                //    {
                //        Message msg{n, std::forward<Args>(args)...};
                //        return { std::move(msg), recv(msg, source, tag).m_handle };

                //    }

                //    /** @brief non-blocking receive which maches any tag from the given source. If a match is found, it
                //      * allocates the container of type Message within this function and returns it in the future.
                //      * The container size will be set according to the matched receive operation.
                //      * @tparam Message a container type
                //      * @tparam Args additional argument types for construction of Message
                //      * @param source source rank
                //      * @param args additional arguments to be passed to new container of type Message at construction 
                //      * @return optional which may hold a future< std::tuple<Message,rank_type,tag_type> > */
                //    template<typename Message, typename... Args>
                //    [[nodiscard]] auto recv_any_tag(rank_type source, Args&& ...args) const
                //    {
                //        return recv_any<Message>(source, MPI_ANY_TAG, std::forward<Args>(args)...);
                //    }

                //    /** @brief non-blocking receive which maches any source using the given tag. If a match is found, it
                //      * allocates the container of type Message within this function and returns it in the future.
                //      * The container size will be set according to the matched receive operation.
                //      * @tparam Message a container type
                //      * @tparam Args additional argument types for construction of Message
                //      * @param tag message tag
                //      * @param args additional arguments to be passed to new container of type Message at construction 
                //      * @return optional which may hold a future< std::tuple<Message,rank_type,tag_type> > */
                //    template<typename Message, typename... Args>
                //    [[nodiscard]] auto recv_any_source(tag_type tag, Args&& ...args) const
                //    {
                //        return recv_any<Message>(MPI_ANY_SOURCE, tag, std::forward<Args>(args)...);
                //    }

                //    /** @brief non-blocking receive which maches any source and any tag. If a match is found, it
                //      * allocates the container of type Message within this function and returns it in the future.
                //      * The container size will be set according to the matched receive operation.
                //      * @tparam Message a container type
                //      * @tparam Args additional argument types for construction of Message
                //      * @param tag message tag
                //      * @param args additional arguments to be passed to new container of type Message at construction 
                //      * @return optional which may hold a future< std::tuple<Message,rank_type,tag_type> > */
                //    template<typename Message, typename... Args>
                //    [[nodiscard]] auto recv_any_source_any_tag(Args&& ...args) const
                //    {
                //        return recv_any<Message>(MPI_ANY_SOURCE, MPI_ANY_TAG, std::forward<Args>(args)...);
                //    }

                //private: // implementation

                //    template<typename Message, typename... Args>
                //    [[nodiscard]] boost::optional< future< std::tuple<Message, rank_type, tag_type> > >
                //    recv_any(rank_type source, tag_type tag, Args&& ...args) const
                //    {
                //        MPI_Message mpi_msg;
                //        status st;
                //        int flag = 0;
                //        GHEX_CHECK_MPI_RESULT(MPI_Improbe(source, tag, *this, &flag, &mpi_msg, &st.get()));
                //        if (flag)
                //        {
                //            int count;
                //            GHEX_CHECK_MPI_RESULT(MPI_Get_count(&st.get(), MPI_CHAR, &count));
                //            Message msg(count/sizeof(typename Message::value_type), std::forward<Args>(args)...);
                //            request req;
                //            GHEX_CHECK_MPI_RESULT(MPI_Imrecv(msg.data(), count, MPI_CHAR, &mpi_msg, &req.get()));
                //            using future_t = future<std::tuple<Message,rank_type,tag_type>>;
                //            return future_t{ std::make_tuple(std::move(msg), st.source(), st.tag()), std::move(req) };
                //        }
                //        return boost::none;
                //    }
                //};

            } // namespace mpi

        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_HPP */

