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
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using threading_std = gridtools::ghex::threads::std_thread::primitives;
using threading_atc = gridtools::ghex::threads::atomic::primitives;
using threading = threading_std;
//using threading = threading_atc;
using context_type = gridtools::ghex::tl::context<transport, threading>;
using comm_type = typename context_type::communicator_type;
using msg_type = typename comm_type::message_type;
using rank_type = typename comm_type::rank_type;
using tag_type = typename comm_type::tag_type;
using future = typename comm_type::template future<void>;

const std::size_t size = 1024;

TEST(context, multi) {
    const int num_threads = 4;
    auto context_ptr_1 = gridtools::ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
    auto& context_1 = *context_ptr_1;
    auto context_ptr_2 = gridtools::ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
    auto& context_2 = *context_ptr_2;

    auto func = [&context_1, &context_2]() {
        auto token_1 = context_1.get_token();
        auto comm_1 = context_1.get_communicator(token_1);
        auto token_2 = context_2.get_token();
        auto comm_2 = context_2.get_communicator(token_2);

        auto msg_1 = comm_1.make_message(size*sizeof(int));
        auto msg_2 = comm_2.make_message(size*sizeof(int));

        if (comm_1.rank() == 0) {
            const int payload_offset = 1+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                *reinterpret_cast<int*>(msg_1.data()+i*sizeof(int)) = i+payload_offset;
        }
        if (comm_2.rank() == 0) {
            const int payload_offset = 2+token_2.id();
            for (unsigned int i=0; i<size; ++i)
                *reinterpret_cast<int*>(msg_2.data()+i*sizeof(int)) = i+payload_offset;
        }

        comm_1.barrier();
        comm_2.barrier();

        future fut_1;
        int counter_1 = 0;
        if (comm_1.rank() == 0) {
            for (rank_type i=1; i<comm_1.size(); ++i)
                comm_1.send(msg_1, i, token_1.id(), [&counter_1](msg_type, rank_type, tag_type) { ++counter_1; });
        }
        else {
            fut_1 = comm_1.recv(msg_1, 0, token_1.id());
        }
        future fut_2;
        int counter_2 = 0;
        if (comm_2.rank() == 0) {
            for (rank_type i=1; i<comm_2.size(); ++i)
                comm_2.send(msg_2, i, token_2.id(), [&counter_2](msg_type, rank_type, tag_type) { ++counter_2; });
        }
        else {
            fut_2 = comm_2.recv(msg_2, 0, token_2.id());
        }


        if (comm_1.rank() == 0)
            while(counter_1 != comm_1.size()-1) { comm_1.progress(); }
        if (comm_2.rank() == 0)
            while(counter_2 != comm_2.size()-1) { comm_2.progress(); }

        if (comm_2.rank() != 0)
            fut_2.wait();
        if (comm_1.rank() != 0)
            fut_1.wait();
        
        // check message
        if (comm_1.rank() != 0) {
            const int payload_offset = 1+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                EXPECT_TRUE(*reinterpret_cast<int*>(msg_1.data()+i*sizeof(int)) == (int)i+payload_offset);
        }
        if (comm_2.rank() != 0) {
            const int payload_offset = 2+token_2.id();
            for (unsigned int i=0; i<size; ++i)
                EXPECT_TRUE(*reinterpret_cast<int*>(msg_2.data()+i*sizeof(int)) == (int)i+payload_offset);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i=0; i<num_threads; ++i)
        threads.push_back(std::thread{func});
    for (auto& t : threads)
        t.join();
}

TEST(context, multi_ordered) {
    const int num_threads = 4;
    auto context_ptr_1 = gridtools::ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
    auto& context_1 = *context_ptr_1;

    auto func = [&context_1]() {
        auto token_1 = context_1.get_token();
        auto comm_1 = context_1.get_communicator(token_1);

        auto msg_1 = comm_1.make_message(size*sizeof(int));
        auto msg_2 = comm_1.make_message(size*sizeof(int));

        if (comm_1.rank() == 0) {
            const int payload_offset = 1+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                *reinterpret_cast<int*>(msg_1.data()+i*sizeof(int)) = i+payload_offset;
        }
        if (comm_1.rank() == 0) {
            const int payload_offset = 2+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                *reinterpret_cast<int*>(msg_2.data()+i*sizeof(int)) = i+payload_offset;
        }

        comm_1.barrier();

        // ordered sends/recvs with same tag should arrive in order

        future fut_1;
        int counter_1 = 0;
        if (comm_1.rank() == 0) {
            for (rank_type i=1; i<comm_1.size(); ++i)
                comm_1.send(msg_1, i, token_1.id(), [&counter_1](msg_type, rank_type, tag_type) { ++counter_1; });
        }
        else {
            fut_1 = comm_1.recv(msg_1, 0, token_1.id());
        }
        future fut_2;
        int counter_2 = 0;
        if (comm_1.rank() == 0) {
            for (rank_type i=1; i<comm_1.size(); ++i)
                comm_1.send(msg_2, i, token_1.id(), [&counter_2](msg_type, rank_type, tag_type) { ++counter_2; });
        }
        else {
            fut_2 = comm_1.recv(msg_2, 0, token_1.id());
        }


        if (comm_1.rank() == 0)
            while(counter_1 != comm_1.size()-1) { comm_1.progress(); }
        if (comm_1.rank() == 0)
            while(counter_2 != comm_1.size()-1) { comm_1.progress(); }

        if (comm_1.rank() != 0)
            fut_1.wait();
        if (comm_1.rank() != 0)
            fut_2.wait();
        
        // check message
        if (comm_1.rank() != 0) {
            const int payload_offset = 1+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                EXPECT_TRUE(*reinterpret_cast<int*>(msg_1.data()+i*sizeof(int)) == (int)i+payload_offset);
        }
        if (comm_1.rank() != 0) {
            const int payload_offset = 2+token_1.id();
            for (unsigned int i=0; i<size; ++i)
                EXPECT_TRUE(*reinterpret_cast<int*>(msg_2.data()+i*sizeof(int)) == (int)i+payload_offset);
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int i=0; i<num_threads; ++i)
        threads.push_back(std::thread{func});
    for (auto& t : threads)
        t.join();
}
