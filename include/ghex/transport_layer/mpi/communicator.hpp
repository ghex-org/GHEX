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
#include "../communicator.hpp"
#include "./communicator_base.hpp"
#include "./future.hpp"
#include "./communicator_traits.hpp"

namespace gridtools {
    
    namespace ghex {

        namespace tl {

            /** Mpi communicator which exposes basic non-blocking transport functionality and 
              * returns futures to await said transports to complete. */
            template<>
            class communicator<mpi_tag>
            : public mpi::communicator_base
            {
            public:
                using transport_type = mpi_tag;
                using base_type      = mpi::communicator_base;
                using address_type   = typename base_type::rank_type;
                using rank_type      = typename base_type::rank_type;
                using size_type      = typename base_type::size_type;
                using tag_type       = typename base_type::tag_type;
                using request        = mpi::request;
                using status         = mpi::status;
                template<typename T>
                using future         = mpi::future<T>;
                using traits         = mpi::communicator_traits;

            public:

                communicator(const traits& t = traits{}) : base_type{t.communicator()} {}
                communicator(const base_type& c) : base_type{c} {}
                communicator(const MPI_Comm& c) : base_type{c} {}
                
                communicator(const communicator&) = default;
                communicator(communicator&&) noexcept = default;

                communicator& operator=(const communicator&) = default;
                communicator& operator=(communicator&&) noexcept = default;

                /** @return address of this process */
                address_type address() const { return rank(); }

            public: // send

                /** @brief non-blocking send
                  * @tparam Message a container type
                  * @param msg source container
                  * @param dest destination rank
                  * @param tag message tag
                  * @return completion handle */
                template<typename Message> 
                [[nodiscard]] future<void> send(const Message& msg, rank_type dest, tag_type tag) const
                {
                    request req;
                    GHEX_CHECK_MPI_RESULT(
                        MPI_Isend(reinterpret_cast<const void*>(msg.data()),sizeof(typename Message::value_type)*msg.size(), 
                                  MPI_BYTE, dest, tag, *this, &req.get())
                    );
                    return req;
                }
            
            public: // recv

                /** @brief non-blocking receive
                  * @tparam Message a container type
                  * @param msg destination container
                  * @param source source rank
                  * @param tag message tag
                  * @return completion handle */
                template<typename Message>
                [[nodiscard]] future<void> recv(Message& msg, rank_type source, tag_type tag) const
                {
                    request req;
                    GHEX_CHECK_MPI_RESULT(
                            MPI_Irecv(reinterpret_cast<void*>(msg.data()),sizeof(typename Message::value_type)*msg.size(), 
                                      MPI_BYTE, source, tag, *this, &req.get()));
                    return req;
                }

                /** @brief non-blocking receive which allocates the container within this function and returns it
                  * in the future 
                  * @tparam Message a container type
                  * @tparam Args additional argument types for construction of Message
                  * @param n number of elements to be received
                  * @param source source rank
                  * @param tag message tag
                  * @param args additional arguments to be passed to new container of type Message at construction 
                  * @return completion handle with message as payload */
                template<typename Message, typename... Args>
                [[nodiscard]] future<Message> recv(int n, rank_type source, tag_type tag, Args&& ...args) const
                {
                    Message msg{n, std::forward<Args>(args)...};
                    return { std::move(msg), recv(msg, source, tag).m_handle };

                }

                /** @brief non-blocking receive which maches any tag from the given source. If a match is found, it
                  * allocates the container of type Message within this function and returns it in the future.
                  * The container size will be set according to the matched receive operation.
                  * @tparam Message a container type
                  * @tparam Args additional argument types for construction of Message
                  * @param source source rank
                  * @param args additional arguments to be passed to new container of type Message at construction 
                  * @return optional which may hold a future< std::tuple<rank_type,tag_type,Message> > */
                template<typename Message, typename... Args>
                [[nodiscard]] auto recv_any_tag(rank_type source, Args&& ...args) const
                {
                    return recv_any<Message>(source, MPI_ANY_TAG, std::forward<Args>(args)...);
                }

                /** @brief non-blocking receive which maches any source using the given tag. If a match is found, it
                  * allocates the container of type Message within this function and returns it in the future.
                  * The container size will be set according to the matched receive operation.
                  * @tparam Message a container type
                  * @tparam Args additional argument types for construction of Message
                  * @param tag message tag
                  * @param args additional arguments to be passed to new container of type Message at construction 
                  * @return optional which may hold a future< std::tuple<rank_type,tag_type,Message> > */
                template<typename Message, typename... Args>
                [[nodiscard]] auto recv_any_source(tag_type tag, Args&& ...args) const
                {
                    return recv_any<Message>(MPI_ANY_SOURCE, tag, std::forward<Args>(args)...);
                }

                /** @brief non-blocking receive which maches any source and any tag. If a match is found, it
                  * allocates the container of type Message within this function and returns it in the future.
                  * The container size will be set according to the matched receive operation.
                  * @tparam Message a container type
                  * @tparam Args additional argument types for construction of Message
                  * @param tag message tag
                  * @param args additional arguments to be passed to new container of type Message at construction 
                  * @return optional which may hold a future< std::tuple<rank_type,tag_type,Message> > */
                template<typename Message, typename... Args>
                [[nodiscard]] auto recv_any_source_any_tag(Args&& ...args) const
                {
                    return recv_any<Message>(MPI_ANY_SOURCE, MPI_ANY_TAG, std::forward<Args>(args)...);
                }

            private: // implementation

                template<typename Message, typename... Args>
                [[nodiscard]] boost::optional< future< std::tuple<rank_type, tag_type, Message> > >
                recv_any(rank_type source, tag_type tag, Args&& ...args) const
                {
                    MPI_Message mpi_msg;
                    status st;
                    int flag = 0;
                    GHEX_CHECK_MPI_RESULT(MPI_Improbe(source, tag, *this, &flag, &mpi_msg, &st.get()));
                    if (flag)
                    {
                        int count;
                        GHEX_CHECK_MPI_RESULT(MPI_Get_count(&st.get(), MPI_CHAR, &count));
                        Message msg(count/sizeof(typename Message::value_type), std::forward<Args>(args)...);
                        request req;
                        GHEX_CHECK_MPI_RESULT(MPI_Imrecv(msg.data(), count, MPI_CHAR, &mpi_msg, &req.get()));
                        using future_t = future<std::tuple<rank_type,tag_type,Message>>;
                        return future_t{ std::make_tuple( st.source(), st.tag(), std::move(msg) ), std::move(req) };
                    }
                    return boost::none;
                }
            };

        } // namespace tl

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_HPP */

