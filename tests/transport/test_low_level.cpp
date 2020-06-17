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

//#define GHEX_TEST_COUNT_ITERATIONS

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using context_type = gridtools::ghex::tl::context<transport>;

#define SIZE 40

/**
 * Simple Send recv on two ranks.
 * P0 sends a message to P1 and receive from P1,
 * P1 sends a message to P0 and receive from P0.
 */

int rank;

template <typename M>
void init_msg(M& msg) {
    int* data = msg.template data<int>();
    for (size_t i = 0; i < msg.size()/sizeof(int); ++i) {
        data[i] = static_cast<int>(i);
    }
}

void init_msg(std::vector<unsigned char>& msg) {
    int c = 0;
    for (size_t i = 0; i < msg.size(); i += 4) {
        *(reinterpret_cast<int*>(&msg[i])) = c++;
    }
}

template <typename M>
bool check_msg(M const& msg) {
    bool ok = true;
    if (rank > 1)
        return ok;

    const int* data = msg.template data<int>();
    for (size_t i = 0; i < msg.size()/sizeof(int); ++i) {
        if ( data[i] != static_cast<int>(i) )
            ok = false;
    }
    return ok;
}

bool check_msg(std::vector<unsigned char> const& msg) {
    bool ok = true;
    if (rank > 1)
        return ok;

    int c = 0;
    for (size_t i = 0; i < msg.size(); i += 4) {
        int value = *(reinterpret_cast<int const*>(&msg[i]));
        if ( value != c++ )
            ok = false;
    }
    return ok;
}

template<typename MsgType, typename Context>
auto test_unidirectional(Context& context) {
    auto comm = context.get_communicator();

    MsgType smsg(SIZE);
    MsgType rmsg(SIZE);
    init_msg(smsg);

    if ( rank == 0 ) {
        comm.send(smsg, 1, 1).get();
    } else {
        auto fut = comm.recv(rmsg, 0, 1);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (!fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif
    }
    return rmsg;
}

template<typename MsgType, typename Context>
auto test_bidirectional(Context& context) {
    auto comm = context.get_communicator();
    using comm_type = std::remove_reference_t<decltype(comm)>;

    MsgType smsg(SIZE);
    MsgType rmsg(SIZE);
    init_msg(smsg);

    typename comm_type::template future<void> rfut;

    if ( rank == 0 ) {
        comm.send(smsg, 1, 1).get();
        rfut = comm.recv(rmsg, 1, 2);
    } else if (rank == 1) {
        comm.send(smsg, 0, 2).get();
        rfut = comm.recv(rmsg, 0, 1);
    }

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
        c++;
#endif
     } while (!rfut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";
#endif

    return rmsg;
}

template<typename MsgType, typename Context>
auto test_unidirectional_cb(Context& context) {
    auto comm = context.get_communicator();

    //using allocator_type  = std::allocator<unsigned char>;
    //using smsg_type       = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type       = std::remove_reference_t<decltype(comm)>;
    using cb_msg_type     = typename comm_type::message_type;

    MsgType smsg(SIZE);
    MsgType rmsg(SIZE);
    init_msg(smsg);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = comm.send(smsg, 1, 1);
        fut.wait();
        auto status = comm.progress();
        EXPECT_EQ(status.num(), 0);
    } else {
        comm.recv(rmsg, 0, 1, [ &arrived](cb_msg_type, int /*src*/, int /* tag */) { arrived = true; });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            comm.progress();
         } while (!arrived);
#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif
        auto status = comm.progress();
        EXPECT_EQ(status.num(), 0);
    }
    return rmsg;
}

template<typename MsgType, typename Context>
auto test_bidirectional_cb(Context& context) {

    auto comm = context.get_communicator();

    // using allocator_type  = std::allocator<unsigned char>;
    // using smsg_type       = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type       = std::remove_reference_t<decltype(comm)>;
    using cb_msg_type     = typename comm_type::message_type;

    MsgType smsg(SIZE);
    MsgType rmsg(SIZE);
    init_msg(smsg);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = comm.send(smsg, 1, 1);
        comm.recv(rmsg, 1, 2, [ &arrived,&rmsg](cb_msg_type, int, int) { arrived = true; });
        fut.wait();
    } else if (rank == 1) {
        auto fut = comm.send(smsg, 0, 2);
        comm.recv(rmsg, 0, 1, [ &arrived,&rmsg](cb_msg_type, int, int) { arrived = true; });
        fut.wait();
    }

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
        c++;
#endif
        comm.progress();
     } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";
#endif

    auto status = comm.progress();
    EXPECT_EQ(status.num(), 0);

    return rmsg;
}


template <typename Test>
bool run_test(Test&& test) {
    bool ok;
    auto msg = test();
    ok = check_msg(msg);
    return ok;
}

TEST(low_level, basic_unidirectional_vector) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    using MsgType = std::vector<unsigned char>;
    auto test_func = [&context]() mutable { return test_unidirectional<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}
TEST(low_level, basic_unidirectional_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_unidirectional<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}
TEST(low_level, basic_unidirectional_shared_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::shared_message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_unidirectional<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}

TEST(low_level, basic_bidirectional_vector) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = std::vector<unsigned char>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_bidirectional<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}
TEST(low_level, basic_bidirectional_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_bidirectional<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}
TEST(low_level, basic_bidirectional_shared_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::shared_message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_bidirectional<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}

TEST(low_level, basic_unidirectional_cb_vector) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = std::vector<unsigned char>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_unidirectional_cb<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}
TEST(low_level, basic_unidirectional_cb_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_unidirectional_cb<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}
TEST(low_level, basic_unidirectional_cb_shared_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::shared_message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_unidirectional_cb<MsgType>(context);};
    if (rank == 1) {
        EXPECT_TRUE(run_test(test_func));
    }
    else if (rank == 0) {
        run_test(test_func);
    }
}

TEST(low_level, basic_bidirectional_cb_vector) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = std::vector<unsigned char>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_bidirectional_cb<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}
TEST(low_level, basic_bidirectional_cb_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable{ return test_bidirectional_cb<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}
TEST(low_level, basic_bidirectional_cb_shared_buffer) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    using MsgType = gridtools::ghex::tl::shared_message_buffer<std::allocator<unsigned char>>;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto test_func = [&context]() mutable { return test_bidirectional_cb<MsgType>(context);};
    if (rank < 2) {
        EXPECT_TRUE(run_test(test_func));
    }
}
