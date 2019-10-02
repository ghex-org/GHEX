/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GHEX_MPI_COMMUNICATOR_HPP
#define GHEX_MPI_COMMUNICATOR_HPP

#include <mpi.h>
#include <tuple>
#include <cassert>
#include <boost/optional.hpp>
//#include "../message.hpp"
#include "../shared_message_buffer.hpp"
#include "./communicator_traits.hpp"
#include "./future.hpp"

namespace gridtools
{
namespace ghex
{
namespace mpi
{

//#ifdef NDEBUG
//#define CHECK_MPI_ERROR(x) x;
//#else
//
//#define CHECK_MPI_ERROR(x) \
//    if (x != MPI_SUCCESS)  \
//        throw std::runtime_error("GHEX Error: MPI Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
//#endif
//
//class communicator;
//
//namespace _impl
//{
//
///** The future returned by the send and receive
//         * operations of a communicator object to check or wait on their status.
//        */
//struct mpi_future
//{
//    MPI_Request m_req;
//
//    mpi_future() = default;
//    mpi_future(MPI_Request req) : m_req{req} {}
//
//    /** Function to wait until the operation completed */
//    void wait()
//    {
//        MPI_Status status;
//        CHECK_MPI_ERROR(MPI_Wait(&m_req, &status));
//    }
//
//    /** Function to test if the operation completed
//             *
//            * @return True if the operation is completed
//            */
//    bool ready()
//    {
//        MPI_Status status;
//        int flag;
//        CHECK_MPI_ERROR(MPI_Test(&m_req, &flag, &status));
//        return flag;
//    }
//
//    /** Cancel the future.
//             *
//            * @return True if the request was successfully canceled
//            */
//    bool cancel()
//    {
//        CHECK_MPI_ERROR(MPI_Cancel(&m_req));
//        MPI_Status st;
//        int flag = false;
//        CHECK_MPI_ERROR(MPI_Wait(&m_req, &st));
//        CHECK_MPI_ERROR(MPI_Test_cancelled(&st, &flag));
//        return flag;
//    }
//
//    private:
//        friend ::gridtools::ghex::mpi::communicator;
//        MPI_Request request() const { return m_req; }
//};
//
//} // namespace _impl

/** Class that provides the functions to send and receive messages. A message
     * is an object with .data() that returns a pointer to `unsigned char`
     * and .size(), with the same behavior of std::vector<unsigned char>.
     * Each message will be sent and received with a tag, bot of type int
     */
class communicator
{
public:
    using future_type = ::gridtools::ghex::tl::mpi::future<void>; //_impl::mpi_future;
    using tag_type = int;
    using rank_type = int;

private:

    ::gridtools::ghex::tl::mpi::communicator_base m_mpi_comm;

public:

    communicator(::gridtools::ghex::tl::mpi::communicator_traits const &ct = ::gridtools::ghex::tl::mpi::communicator_traits{}) 
    : m_mpi_comm{ct.communicator()} 
    {}

    rank_type rank() const noexcept { return m_mpi_comm.rank(); }
    rank_type size() const noexcept { return m_mpi_comm.size(); }

    operator MPI_Comm() const { return m_mpi_comm; }

    /** Send a message to a destination with the given tag.
         * It returns a future that can be used to check when the message is available
         * again for the user.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         *
         * @return A future that will be ready when the message can be reused (e.g., filled with new data to send)
         */
    template <typename MsgType>
    [[nodiscard]] future_type send(MsgType const &msg, rank_type dst, tag_type tag) const {
        ::gridtools::ghex::tl::mpi::request req;
        GHEX_CHECK_MPI_RESULT(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req.get()));
        return req;
    }

    /** Send a message to a destination with the given tag. This function blocks until the message has been sent and
         * the message ready to be reused
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         */
    template <typename MsgType>
    void blocking_send(MsgType const &msg, rank_type dst, tag_type tag) const
    {
        GHEX_CHECK_MPI_RESULT(MPI_Send(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm));
    }

    /** Receive a message from a destination with the given tag.
         * It returns a future that can be used to check when the message is available
         * to be read.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         *
         * @param msg Const reference to a message that will contain the data
         * @param src Source of the message
         * @param tag Tag associated with the message
         *
         * @return A future that will be ready when the message can be read
         */
    template <typename MsgType>
    [[nodiscard]] future_type recv(MsgType &msg, rank_type src, tag_type tag) const {
        ::gridtools::ghex::tl::mpi::request req;
        GHEX_CHECK_MPI_RESULT(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &req.get()));
        return req;
    }

    /** Receive a message from any destination with any tag.
         * In case of success the function returns a tuple of rank, tag and message that has been received. The
         * necessary storage will be allocated using a shared_message<Allocator>.
         * It is safe to call this function in a multi-threaded mpi environment, however, there is no way to prevent
         * one thread receiving a message intended for another thread.
         *
         * @tparam Allocator Allocator type used for creating a shared_message.
         * @param alloc Allocator instance
         * @return Optional which may hold a tuple of rank, tag and message
         */
    template < typename Allocator = std::allocator<unsigned char> >
    //boost::optional<std::tuple<rank_type,tag_type,shared_message<Allocator>>> 
    boost::optional<std::tuple<rank_type,tag_type,::gridtools::ghex::tl::shared_message_buffer<Allocator>>> 
    recv_any(Allocator alloc = Allocator{}) const 
    {
        MPI_Message mpi_msg;
        MPI_Status st;
        int flag = 0;
        GHEX_CHECK_MPI_RESULT(MPI_Improbe(MPI_ANY_SOURCE, MPI_ANY_TAG, m_mpi_comm, &flag, &mpi_msg, &st));
        if (flag)
        {
            //shared_message<Allocator> msg(alloc);
            ::gridtools::ghex::tl::shared_message_buffer<Allocator> msg(alloc);
            int count;
            MPI_Get_count(&st, MPI_CHAR, &count);
            msg.reserve(count);
            msg.resize(count);
            rank_type r = st.MPI_SOURCE;
            tag_type t = st.MPI_TAG;
            GHEX_CHECK_MPI_RESULT(MPI_Mrecv(msg.data(), count, MPI_CHAR, &mpi_msg, MPI_STATUS_IGNORE));
            return std::make_tuple(r,t,msg);
        }
        return boost::none;
    }
};

} //namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
