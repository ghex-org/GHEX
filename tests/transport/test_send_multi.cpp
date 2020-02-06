#include <ghex/threads/none/primitives.hpp>
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

using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

//#define GHEX_TEST_COUNT_ITERATIONS
//
const int SIZE = 4000000;
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

TEST(transport, send_multi) {
    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        EXPECT_EQ(size, 4);
    }

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    comm.barrier();

    rank = context.rank();

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;

    if (rank == 0) {
        smsg_type smsg{SIZE};
        init_msg(smsg);

        std::array<int, 3> dsts = {1,2,3};

        auto fut_vec = comm.send_multi(smsg, dsts, 42);

        for (auto& fut : fut_vec)
            fut.wait();
    }
    else {
        smsg_type rmsg{SIZE};
        comm.recv(rmsg, 0, 42).wait();
        bool ok = check_msg(rmsg);

        EXPECT_TRUE(ok);
    }

    auto status = comm.progress();
    EXPECT_EQ(status.num(), 0);
}

TEST(transport, send_multi_cb) {
    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        EXPECT_EQ(size, 4);
    }

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    comm.barrier();

    rank = context.rank();

    using comm_type      = std::remove_reference_t<decltype(comm)>;
    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    //using smsg_type      = gridtools::ghex::tl::message_buffer<allocator_type>;
    using cb_msg_type    = comm_type::message_type;
    using rank_type      = comm_type::rank_type;
    using tag_type       = comm_type::tag_type;

    if (rank == 0) {

        smsg_type smsg{SIZE};
        init_msg(smsg);

        std::array<int, 3> dsts = {1,2,3};

        bool arrived = false;
        auto req_vec = comm.send_multi(smsg, dsts, 42, [&arrived](cb_msg_type, rank_type, tag_type){ arrived=true;});

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            comm.progress();
         } while (!arrived);

        EXPECT_EQ(smsg.use_count(), 1);
        for (auto& req : req_vec)
            EXPECT_TRUE(req.test());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout  << "\n***********\n";
        std::cout  <<   "*" << std::setw(8) << c << " *\n";
        std::cout  << "***********\n";
#endif

    } else {
        smsg_type rmsg{SIZE};
        comm.recv(rmsg, 0, 42).wait();
        bool ok = check_msg(rmsg);

        EXPECT_TRUE(ok);
    }

    auto status = comm.progress();
    EXPECT_EQ(status.num(), 0);
}

TEST(transport, send_multi_cb_move) {
    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        EXPECT_EQ(size, 4);
    }

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    comm.barrier();

    rank = context.rank();

    using comm_type      = std::remove_reference_t<decltype(comm)>;
    using allocator_type = std::allocator<unsigned char>;
    //using smsg_type      = gridtools::ghex::tl::message_buffer<allocator_type>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using cb_msg_type    = comm_type::message_type;
    using rank_type      = comm_type::rank_type;
    using tag_type       = comm_type::tag_type;

    if (rank == 0) {

        smsg_type smsg{SIZE};
        init_msg(smsg);

        std::array<int, 3> dsts = {1,2,3};

        bool arrived = false;
        auto req_vec = comm.send_multi(std::move(smsg), dsts, 42, [&arrived](cb_msg_type, rank_type, tag_type){ arrived=true;});

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            comm.progress();
         } while (!arrived);

        EXPECT_EQ(smsg.use_count(), 0);
        for (auto& req : req_vec)
            EXPECT_TRUE(req.test());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout  << "\n***********\n";
        std::cout  <<   "*" << std::setw(8) << c << " *\n";
        std::cout  << "***********\n";
#endif

    } else {
        smsg_type rmsg{SIZE};
        comm.recv(rmsg, 0, 42).wait();
        bool ok = check_msg(rmsg);

        EXPECT_TRUE(ok);
    }

    auto status = comm.progress();
    EXPECT_EQ(status.num(), 0);
}
