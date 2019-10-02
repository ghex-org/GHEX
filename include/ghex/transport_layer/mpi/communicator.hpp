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
#ifndef INCLUDED_MPI_HPP
#define INCLUDED_MPI_HPP

#include <boost/optional.hpp>
#include "../communicator.hpp"
#include "./communicator_base.hpp"
#include "./future.hpp"
#include "./communicator_traits.hpp"

namespace gridtools {
    
    namespace ghex {

        namespace tl {

            /** @brief mpi communicator */
            template<>
            class communicator<mpi_tag>
            {
            public:
                using protocol_type = mpi_tag;
                using base_type     = mpi::communicator_base;
                using address_type  = typename base_type::rank_type;
                using rank_type     = typename base_type::rank_type;
                using size_type     = typename base_type::size_type;
                using tag_type      = typename base_type::tag_type;
                using request       = mpi::request;
                using status        = mpi::status;
                template<typename T>
                using future        = mpi::future<T>;
                using traits        = mpi::communicator_traits;

                base_type m_comm;

                operator const MPI_Comm&() const noexcept { return m_comm; }
                operator       MPI_Comm&()       noexcept { return m_comm; }

            public:

                communicator(const traits& t = traits{}) : m_comm{t.communicator()} {}
                communicator(const base_type& c) : m_comm{c} {}
                communicator(const MPI_Comm& c) : m_comm{c} {}
                
                communicator(const communicator&) = default;
                communicator(communicator&&) = default;

                communicator& operator=(const communicator&) = default;
                communicator& operator=(communicator&&) = default;

                /** @return address of this process */
                address_type address() const { return m_comm.rank(); }
                
                /** @return rank of this process */
                rank_type rank() const { return m_comm.rank(); }

                /** @return size of communicator group*/
                size_type size() const { return m_comm.size(); }

                void barrier() { m_comm.barrier(); }

            public: // send

                /**
                 * @brief non-blocking send
                 * @tparam T data type
                 * @param dest destination rank
                 * @param tag message tag
                 * @param buffer pointer to source buffer
                 * @param n number of elements in buffer
                 * @return completion handle
                 */
                template<typename T>
                [[nodiscard]] future<void> send(rank_type dest, tag_type tag, const T* buffer, int n) const
                {
                    request req;
                    GHEX_CHECK_MPI_RESULT(MPI_Isend(reinterpret_cast<const void*>(buffer),sizeof(T)*n, MPI_BYTE, dest, tag, m_comm, &req.get()));
                    return req;
                }

                /**
                 * @brief non-blocking send (vector interface)
                 * @tparam T data type
                 * @tparam Vector vector type (contiguous memory)
                 * @tparam Allocator allocator type
                 * @param dest destination rank
                 * @param tag message tag
                 * @param vec source vector
                 * @return completion handle
                 */
                template<typename Message> 
                [[nodiscard]] future<void> send(rank_type dest, tag_type tag, const Message& msg) const
                {
                    return send(dest, tag, msg.data(), msg.size()); 
                }
            
            public: // recv

                /**
                 * @brief non-blocking receive
                 * @tparam T data type
                 * @param source source rank
                 * @param tag message tag
                 * @param buffer pointer destination buffer
                 * @param n number of elements in buffer
                 * @return completion handle
                 */
                template<typename T>
                [[nodiscard]] future<void> recv(rank_type source, tag_type tag, T* buffer, int n) const
                {
                    request req;
                    GHEX_CHECK_MPI_RESULT(MPI_Irecv(reinterpret_cast<void*>(buffer),sizeof(T)*n, MPI_BYTE, source, tag, m_comm, &req.get()));
                    return req;
                }

                template<typename Message>
                [[nodiscard]] future<void> recv(rank_type source, tag_type tag, Message& msg) const
                {
                    return recv(source, tag, msg.data(), msg.size());
                }

                template<typename Message, typename... Args>
                [[nodiscard]] future<Message> recv(rank_type source, tag_type tag, int n, Args&& ...args) const
                {
                    Message msg{n, std::forward<Args>(args)...};
                    return { std::move(msg), recv(source, tag, msg.data(), msg.size()).m_handle };

                }

                template<typename Message, typename... Args>
                [[nodiscard]] auto recv_any_tag(rank_type source, Args&& ...args) const
                {
                    return recv_any<Message>(source, MPI_ANY_TAG, std::forward<Args>(args)...);
                }

                template<typename Message, typename... Args>
                [[nodiscard]] auto recv_any_source(tag_type tag, Args&& ...args) const
                {
                    return recv_any<Message>(MPI_ANY_SOURCE, tag, std::forward<Args>(args)...);
                }

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
                    GHEX_CHECK_MPI_RESULT(MPI_Improbe(source, tag, m_comm, &flag, &mpi_msg, &st.get()));
                    if (flag)
                    {
                        int count;
                        GHEX_CHECK_MPI_RESULT(MPI_Get_count(&st.get(), MPI_CHAR, &count));
                        Message msg(count, std::forward<Args>(args)...);
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

#endif /* INCLUDED_MPI_HPP */

