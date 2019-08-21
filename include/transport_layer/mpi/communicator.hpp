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

#include <iostream>
#include <mpi.h>
#include <future>
#include <functional>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include "./message.hpp"
#include "./communicator_traits.hpp"
#include <algorithm>
#include <deque>
#include <boost/optional.hpp>

namespace gridtools
{
namespace ghex
{
namespace mpi
{

#ifdef NDEBUG
#define CHECK_MPI_ERROR(x) x;
#else

#define CHECK_MPI_ERROR(x) \
    if (x != MPI_SUCCESS)  \
        throw std::runtime_error("GHEX Error: MPI Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
#endif

class communicator;

namespace _impl
{
/** The future returned by the send and receive
         * operations of a communicator object to check or wait on their status.
        */
struct mpi_future
{
    MPI_Request m_req;

    mpi_future() = default;
    mpi_future(MPI_Request req) : m_req{req} {}

    /** Function to wait until the operation completed */
    void wait()
    {
        MPI_Status status;
        CHECK_MPI_ERROR(MPI_Wait(&m_req, &status));
    }

    /** Function to test if the operation completed
             *
            * @return True if the operation is completed
            */
    bool ready()
    {
        MPI_Status status;
        int flag;
        CHECK_MPI_ERROR(MPI_Test(&m_req, &flag, &status));
        return flag;
    }

    /** Cancel the future.
             *
            * @return True if the request was successfully canceled
            */
    bool cancel()
    {
        CHECK_MPI_ERROR(MPI_Cancel(&m_req));
        MPI_Status st;
        int flag = false;
        CHECK_MPI_ERROR(MPI_Wait(&m_req, &st));
        CHECK_MPI_ERROR(MPI_Test_cancelled(&st, &flag));
        return flag;
    }

    private:
        friend ::gridtools::ghex::mpi::communicator;
        MPI_Request request() const { return m_req; }
};

class cb_request_t
{
    MPI_Request m_req;

public:
    cb_request_t(MPI_Request r)
        : m_req{r}
    {
    }

    cb_request_t() : m_req{0} {}

    cb_request_t(cb_request_t const &) = default;
    cb_request_t(cb_request_t &&) = default;

    cb_request_t &operator=(cb_request_t const &oth)
    {
        m_req = oth.m_req;
        return *this;
    }

protected:
    friend class ::gridtools::ghex::mpi::communicator;
    MPI_Request operator()() const { return m_req; }
};

} // namespace _impl

/** Class that provides the functions to send and receive messages. A message
     * is an object with .data() that returns a pointer to `unsigned char`
     * and .size(), with the same behavior of std::vector<unsigned char>.
     * Each message will be sent and received with a tag, bot of type int
     */
class communicator
{
public:
    using future_type = _impl::mpi_future;
    using request_type = _impl::cb_request_t;

private:
    using tag_type = int;
    using rank_type = int;

    template<typename Msg>
    struct call_back_
    {
        Msg m_msg;
        std::function<void(rank_type, tag_type, Msg&&)> m_inner_cb;

        template<typename Callback>
        call_back_(Msg&& msg, Callback&& cb)
        : m_msg(std::move(msg)), m_inner_cb(std::forward<Callback>(cb))
        {}

        // dangerous ahead:
        call_back_(const call_back_& x)
        : m_msg(std::move(x.m_msg)),
          m_inner_cb(std::move(x.m_inner_cb))
        {}

        call_back_(call_back_&&) = default;

        call_back_& operator=(call_back_&&) = default;

        void operator()(rank_type r, tag_type t)
        {
            m_inner_cb(r,t,std::move(m_msg));
        }

        Msg& message() { return m_msg; }
    };

public:
    using element_t = std::tuple<std::function<void(rank_type, tag_type)>, rank_type, tag_type, future_type>;
    using cb_container_t = std::deque<element_t>;
    std::array<cb_container_t,2> m_callbacks;
    MPI_Comm m_mpi_comm;

public:

    communicator(communicator_traits const &ct = communicator_traits{}) : m_mpi_comm{ct.communicator()} {}

    ~communicator()
    {
        if (m_callbacks[0].size() != 0 || m_callbacks[1].size() != 0)
        {
            std::terminate();
        }
    }

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
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        return req;
    }

    /** Send a message to a destination with the given tag. When the message is sent, and
         * the message ready to be reused, the given call-back is invoked with the destination
         *  and tag of the message sent.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         * @tparam CallBack Funciton to call when the message has been sent and the message ready for reuse
         *
         * @param msg Const reference to a message to send
         * @param dst Destination of the message
         * @param tag Tag associated with the message
         * @param cb  Call-back function with signature void(int, int)
         *
         */
    template <typename MsgType, typename CallBack>
    //std::enable_if_t<!std::is_reference<MsgType>::value>
    void 
    send(MsgType const &msg, rank_type dst, tag_type tag, CallBack &&cb)
    {
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        m_callbacks[0].push_back( std::make_tuple(std::forward<CallBack>(cb), dst, tag, future_type(req)) );
    }

    /*template <typename MsgType, typename CallBack>
    std::enable_if_t<std::is_rvalue_reference<MsgType&&>::value>
    //void 
    send(MsgType&& msg, rank_type dst, tag_type tag, CallBack &&cb)
    {
        call_back_<MsgType> cb2(std::move(msg), std::forward<CallBack>(cb));
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Isend(cb.message().data(), cb.message().size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        m_callbacks.emplace(std::make_pair(req, std::make_tuple(std::move(cb2), dst, tag)));
    }*/

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
        CHECK_MPI_ERROR(MPI_Send(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm));
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
        MPI_Request request;
        CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));
        return request;
    }

    /** Receive a message from a source with the given tag. When the message arrives, and
         * the message ready to be read, the given call-back is invoked with the source
         *  and tag of the message sent.
         *
         * @tparam MsgType message type (this could be a std::vector<unsigned char> or a message found in message.hpp)
         * @tparam CallBack Funciton to call when the message has been sent and the message ready to be read
         *
         * @param msg rvalue reference to a message that will contain the data
         * @param src Source of the message
         * @param tag Tag associated with the message
         * @param cb  Call-back function with signature void(int, int, MsgType&&)
         *
         */
    template <typename MsgType, typename CallBack>
    std::enable_if_t<std::is_rvalue_reference<MsgType&&>::value>
    recv(MsgType&& msg, rank_type src, tag_type tag, CallBack &&cb)
    {
        call_back_<MsgType> cb2(std::move(msg), std::forward<CallBack>(cb));
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Irecv(cb2.message().data(), cb2.message().size(), MPI_BYTE, src, tag, m_mpi_comm, &req));
        m_callbacks[1].push_back( std::make_tuple(std::move(cb2), src, tag, future_type(req)) );
    }

    /** Send a message (shared_message type) to a set of destinations listed in
         * a container, with a same tag.
         *
         * @tparam Allc   Allocator of the dshared_message (deduced)
         * @tparam Neighs Container with the neighbor IDs
         *
         * @param msg    The message to send (must be shared_message<Allc> type
         * @param neighs Container of the IDs of the recipients
         * @param tag    Tag of the message
         */
    template <typename Allc, typename Neighs>
    void send_multi(shared_message<Allc> &msg, Neighs const &neighs, int tag)
    {
        for (auto id : neighs)
        {
            auto keep_message = [msg](int, int) {
                /*if (rank == 0) std::cout  << "KM DST " << p << ", TAG " << t << " USE COUNT " << msg.use_count() << "\n";*/
            };
            send(msg, id, tag, std::move(keep_message));
        }
    }

    /** Function to invoke to poll the transport layer and check for the completions
         * of the operations without a future associated to them (that is, they are associated
         * to a call-back). When an operation completes, the corresponfing call-back is invoked
         * with the rank and tag associated with that request.
         *
         * @return bool True if there are pending requests, false otherwise
         */
    bool progress()
    {

        for (auto& cb_container : m_callbacks) 
        {
            const unsigned int size = cb_container.size();
            for (unsigned int i=0; i<size; ++i) 
            {
                element_t element = std::move(cb_container.front());
                cb_container.pop_front();

                if (std::get<3>(element).ready())
                {
                    auto f = std::move(std::get<0>(element));
                    auto x = std::get<1>(element);
                    auto y = std::get<2>(element);
                    f(x, y);
                    break;
                }
                else
                {
                    cb_container.push_back(std::move(element));
                }
            }
        }
        return !(m_callbacks[0].size() == 0 && m_callbacks[1].size()==0);
    }


    boost::optional<future_type> detach_send(rank_type rank, tag_type tag) {
        return detach(m_callbacks[0], rank, tag);
    }

    boost::optional<future_type> detach_recv(rank_type rank, tag_type tag) {
        return detach(m_callbacks[1], rank, tag);
    }
    
    boost::optional<future_type> detach(cb_container_t& cb_container, rank_type rank, tag_type tag)
    {
        auto it = std::find_if(
            cb_container.begin(), 
            cb_container.end(), 
            [rank, tag](auto const& x) 
            {
                return (std::get<1>(x) == rank && std::get<2>(x) == tag);
            });

        if (it != cb_container.end()) 
        {
            auto fut = std::move(std::get<3>(*it));
            cb_container.erase(it);
            return fut;
        } else 
            return boost::none;
    }

    /*template <typename Callback>
    void attach(future_type const& fut, rank_type rank, tag_type tag, Callback&& cb) {

        auto it = std::find_if(m_callbacks.begin(), m_callbacks.end(),
            [rank, tag](auto const& x) {
                if (std::get<1>(x.second) == rank && std::get<2>(x.second) == tag) {
                    return true;
                }
                return false;
            });

        if (it == m_callbacks.end()) {
            m_callbacks.emplace(std::make_pair(fut.request(), std::make_tuple(std::forward<Callback>(cb), rank, tag)));
        } else {
            throw std::runtime_error("GHEX ERROR: There is already such a request in line");
        }
    }*/

    /**
         * @brief Function to cancel all pending requests for send/recv with callbacks.
         * This cancel also the calls to `send_multi`. Canceling is an expensive operation
         * and should be used in exceptional cases.
         *
         * @return True if all the pending requests are canceled, false otherwise.
         */
    bool cancel_callbacks()
    {
        bool result = cancel_callbacks(m_callbacks[0]);
        result = result && cancel_callbacks(m_callbacks[1]); 

        return result;
    }

    bool cancel_callbacks(cb_container_t& cb_container)
    {
        int result = true;
        const unsigned int size = cb_container.size();
        for (unsigned int i=0; i<size; ++i) 
        {
            element_t element = std::move(cb_container.front());
            cb_container.pop_front();
            auto& fut = std::get<3>(element);
            if (!fut.ready())
            {
            MPI_Request r = fut.m_req;
            CHECK_MPI_ERROR(MPI_Cancel(&r));
            MPI_Status st;
            int flag = false;
            CHECK_MPI_ERROR(MPI_Wait(&r, &st));
            CHECK_MPI_ERROR(MPI_Test_cancelled(&st, &flag));
            if (!flag) std::cout << "  not cancelled!!!" << std::endl;
            result &= flag;
            }
        }
        return result;
    }


};

} //namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
