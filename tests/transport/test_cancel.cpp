/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#include <vector>
#include <iomanip>
#include <utility>

#include <gtest/gtest.h>

#include <ghex/threads/none/primitives.hpp>
#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using threading = gridtools::ghex::threads::none::primitives;

template<typename Message, typename Context>
bool test_1(Context& context, unsigned int size) {
    auto comm = context.get_communicator(context.get_token());
    EXPECT_TRUE(comm.size() == 4);
    Message msg(size*sizeof(int));

    if (comm.rank() == 0) {
        int* data = reinterpret_cast<int*>(msg.data());
        for (unsigned int i = 0; i < size; ++i)
            data[i] = i;
        std::array<int, 3> dsts = {1,2,3};
        auto futures = comm.send_multi(msg, dsts, 42+42);
        for (auto& fut : futures)
            fut.wait();
        comm.barrier();
        return true;
    }
    else {
        auto fut = comm.recv(msg, 0, 42);
        bool ok = fut.cancel();
        comm.recv(msg, 0, 42+42).wait();
        comm.barrier();
        return ok;
    }
}

TEST(cancel, future) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    EXPECT_TRUE(test_1<std::vector<unsigned char>>(context,1));
    EXPECT_TRUE(test_1<std::vector<unsigned char>>(context,32));
    EXPECT_TRUE(test_1<std::vector<unsigned char>>(context,4096));
}

template<typename Message, typename Context>
bool test_2(Context& context, unsigned int size) {
    auto comm = context.get_communicator(context.get_token());
    EXPECT_TRUE(comm.size() == 4);
    Message msg(size*sizeof(int));

    if (comm.rank() == 0) {
        int* data = reinterpret_cast<int*>(msg.data());
        for (unsigned int i = 0; i < size; ++i)
            data[i] = i;
        std::array<int, 3> dsts = {1,2,3};
        auto futures = comm.send_multi(msg, dsts, 42+42);
        for (auto& fut : futures)
            fut.wait();
        comm.barrier();
        return true;
    }
    else {
        using comm_t = typename Context::communicator_type;
        using msg_t = typename comm_t::message_type;
        using rank_t = typename comm_t::rank_type;
        using tag_t = typename comm_t::tag_type;

        int counter = 0;
        auto req = comm.recv(msg, 0, 42, [&counter](msg_t, rank_t, tag_t){ ++counter; });
        bool ok = req.cancel();
        auto status = comm.progress();
        while (status.num_cancels() == 0) { status += comm.progress(); };
        comm.recv(msg, 0, 42+42).wait();
        EXPECT_TRUE(counter == 0);
        comm.barrier();
        return ok;
    }
}

TEST(cancel, callbacks) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    EXPECT_TRUE(test_2<std::vector<unsigned char>>(context,1));
    EXPECT_TRUE(test_2<std::vector<unsigned char>>(context,32));
    EXPECT_TRUE(test_2<std::vector<unsigned char>>(context,4096));
}
