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
#include <algorithm>
#include <deque>
#include <string>

#include "threads.hpp"

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
    using tag_type = int;
    using rank_type = int;

public:
    MPI_Comm m_mpi_comm;

    static rank_type m_rank;
    static rank_type m_size;

    rank_type m_thrid;
    rank_type m_nthr;

    static const std::string name;
    
public:

    void whoami(){
	printf("I am %d/%d:%d/%d\n", m_rank, m_size, m_thrid, m_nthr);
    }

    /*
      Has to be called at in the begining of the parallel region.
     */
    void init_mt(){
	m_thrid = GET_THREAD_NUM();
	m_nthr = GET_NUM_THREADS();
	printf("create communicator %d:%d/%d pointer %x\n", m_rank, m_thrid, m_nthr, this);

	/* duplicate the communicator - all threads in order: this is a collective! */
	for(int tid=0; tid<m_nthr; tid++){
	    if(m_thrid==tid) {
		MPI_Comm_dup(MPI_COMM_WORLD, &m_mpi_comm);
	    }
#pragma omp barrier
	}
	MPI_Barrier(m_mpi_comm);
    }

    communicator()
    {
	int mode;

	if(!IN_PARALLEL()){
#ifdef THREAD_MODE_MULTIPLE
	    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
#else
	    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
	    MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &m_size);
	    m_mpi_comm = MPI_COMM_WORLD;
	}
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

    template <typename MsgType>
    [[nodiscard]] future_type recv(MsgType &msg, rank_type src, tag_type tag) const {
        MPI_Request request;
        CHECK_MPI_ERROR(MPI_Irecv(msg.data(), msg.size(), MPI_BYTE, src, tag, m_mpi_comm, &request));
        return request;
    }

    void fence()
    {
	MPI_Barrier(m_mpi_comm);
    }

};

/** this has to be here, because the class needs to be complete */
extern communicator comm;
DECLARE_THREAD_PRIVATE(comm)
communicator comm;

const std::string communicator::name = "ghex::mpi";
communicator::rank_type communicator::m_rank;
communicator::rank_type communicator::m_size;

} //namespace mpi
} // namespace ghex
} // namespace gridtools

#endif
