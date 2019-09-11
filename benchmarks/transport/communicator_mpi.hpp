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
#include <algorithm>
#include <deque>
#include <string>

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
    MPI_Request m_req = MPI_REQUEST_NULL;

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

private:
    using tag_type = int;
    using rank_type = int;

    template<typename Msg>
    struct call_back_owning   // TODO: non-owning is faster :(
    {
        Msg m_msg;
	std::function<void(rank_type, tag_type, Msg&)> m_inner_cb;

        template<typename Callback>
        call_back_owning(Msg& msg, Callback&& cb)
	    : m_msg(msg), m_inner_cb(std::forward<Callback>(cb))
        {}
    
        template<typename Callback>
        call_back_owning(Msg&& msg, Callback&& cb)
	    : m_msg(std::move(msg)), m_inner_cb(std::forward<Callback>(cb))
        {}
    
        call_back_owning(const call_back_owning& x) = default;
        call_back_owning(call_back_owning&&) = default;

        void operator()(rank_type r, tag_type t)
        {
            m_inner_cb(r,t,m_msg);
        }
    
        Msg& message() { return m_msg; }
    };

public:
    using element_t = std::tuple<std::function<void(rank_type, tag_type)>, rank_type, tag_type, future_type>;
    using cb_container_t = std::deque<element_t>;
    std::array<cb_container_t,2> m_callbacks;
    MPI_Comm m_mpi_comm;
    rank_type m_rank, m_size;

    static const std::string name;
    
public:

    communicator()
    {
	int mode;
#ifdef THREAD_MODE_MULTIPLE
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
#else
	// MPI_Init(NULL, NULL);
	MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
	MPI_Comm_dup(MPI_COMM_WORLD, &m_mpi_comm);
	MPI_Comm_rank(m_mpi_comm, &m_rank);
	MPI_Comm_size(m_mpi_comm, &m_size);    
    }

    ~communicator()
    {
	// no good for benchmarks
        // if (m_callbacks[0].size() != 0 || m_callbacks[1].size() != 0)
        // {
        //     std::terminate();
        // }
	MPI_Finalize();
    }

    template <typename MsgType>
    [[nodiscard]] future_type send(MsgType const &msg, rank_type dst, tag_type tag) const {
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Isend(msg.data(), msg.size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        return req;
    }

    template <typename MsgType, typename CallBack>
    void 
    send(MsgType &msg, rank_type dst, tag_type tag, CallBack &&cb)
    {
        call_back_owning<MsgType> cb2(msg, std::forward<CallBack>(cb));
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Isend(cb2.message().data(), cb2.message().size(), MPI_BYTE, dst, tag, m_mpi_comm, &req));
        m_callbacks[0].push_back( std::make_tuple(std::move(cb2), dst, tag, future_type(req)) );
    }

    template <typename MsgType>
    [[nodiscard]] future_type recv(MsgType &msg, rank_type src, tag_type tag) const {
        MPI_Request request;
        CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));
        return request;
    }

    // uses msg reference (user is not allowed to delete it)
    template <typename MsgType, typename CallBack>
    void
    recv(MsgType& msg, rank_type src, tag_type tag, CallBack &&cb)
    {
        call_back_owning<MsgType> cb2(msg, std::forward<CallBack>(cb));
        MPI_Request req;
        CHECK_MPI_ERROR(MPI_Irecv(cb2.message().data(), cb2.message().size(), MPI_BYTE, src, tag, m_mpi_comm, &req));
        m_callbacks[1].push_back( std::make_tuple(std::move(cb2), src, tag, future_type(req)) );
    }

    unsigned progress()
    {
	int completed = 0;
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
		    completed++;
                }
                else
                {
                    cb_container.push_back(std::move(element));
                }
            }
        }
        return completed;
    }

    void fence()
    {
	MPI_Barrier(m_mpi_comm);
    }

};

const std::string communicator::name = "ghex::mpi";

} //namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
