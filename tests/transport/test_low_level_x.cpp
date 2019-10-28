#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>
#include <utility>

#include <gtest/gtest.h>

/**
 * Simple Send recv on two ranks.
 * P0 sends a message to P1 and receive from P1,
 * P1 sends a message to P0 and receive from P0.
 */

int rank;

auto test1() {
    using comm_type = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
    comm_type sr;

    std::vector<unsigned char> smsg = {0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0,8,0,0,0,9,0,0,0};
    std::vector<unsigned char> rmsg(40, 40);

    comm_type::future<void> rfut;

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1).get();
        rfut = sr.recv(rmsg, 1, 2);
    } else if (rank == 1) {
        sr.send(smsg, 0, 2).get();
        rfut = sr.recv(rmsg, 0, 1);
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

auto test2() {
    using sr_comm_type   = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using cb_comm_type   = gridtools::ghex::tl::callback_communicator<sr_comm_type,allocator_type>;

    sr_comm_type sr;
    cb_comm_type cb_comm(sr);

    std::vector<unsigned char> smsg = {0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0,8,0,0,0,9,0,0,0};
    smsg_type rmsg(40);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        cb_comm.recv(rmsg, 1, 2, [ &arrived,&rmsg](const smsg_type&, int, int) { arrived = true; });
        fut.wait();
    } else if (rank == 1) {
        auto fut = sr.send(smsg, 0, 2);
        cb_comm.recv(rmsg, 0, 1, [ &arrived,&rmsg](const smsg_type&, int, int) { arrived = true; });
        fut.wait();
    }

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
        c++;
#endif
        cb_comm.progress();
     } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";
#endif

    EXPECT_FALSE(cb_comm.progress());

    return rmsg;
}

auto test1_mesg() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    gridtools::ghex::tl::message_buffer<> smsg{40};

    int* data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::tl::message_buffer<> rmsg{40};

    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>::future<void> rfut;

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1).get();
        rfut = sr.recv(rmsg, 1, 2);
    } else if (rank == 1) {
        sr.send(smsg, 0, 2).get();
        rfut = sr.recv(rmsg, 0, 1);
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

auto test2_mesg() {
    using sr_comm_type   = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using cb_comm_type   = gridtools::ghex::tl::callback_communicator<sr_comm_type,allocator_type>;

    sr_comm_type sr;
    cb_comm_type cb_comm(sr);

    gridtools::ghex::tl::message_buffer<> smsg{40};

    int* data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    smsg_type rmsg{40};

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        cb_comm.recv(rmsg, 1, 2, [ &arrived](const smsg_type&, int, int) { arrived = true; });
        fut.wait();
    } else if (rank == 1) {
        auto fut = sr.send(smsg, 0, 2);
        cb_comm.recv(rmsg, 0, 1, [ &arrived](const smsg_type&, int, int) { arrived = true; });
        fut.wait();
    }

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
        c++;
#endif
        cb_comm.progress();
     } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";
#endif

    EXPECT_FALSE(cb_comm.progress());


    return rmsg;
}

auto test1_shared_mesg() {
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> sr;

    gridtools::ghex::tl::shared_message_buffer<> smsg{40};
    int* data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::tl::shared_message_buffer<> rmsg{40};

    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>::future<void> rfut;

    if ( rank == 0 ) {
        auto sf = sr.send(smsg, 1, 1);
        rfut = sr.recv(rmsg, 1, 2);
        sf.wait();
    } else if (rank == 1) {
        auto sf = sr.send(smsg, 0, 2);
        rfut = sr.recv(rmsg, 0, 1);
        sf.wait();
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

template <typename Test>
bool run_test(Test&& test) {
    bool ok;
    auto msg = test();


    ok = check_msg(msg);
    return ok;
}


TEST(low_level, basic_x) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < 2) {
        EXPECT_TRUE(run_test(test1));
    }
}

TEST(low_level, basic_x_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < 2) {
        EXPECT_TRUE(run_test(test2));
    }
}

TEST(low_level, basic_x_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < 2) {
        EXPECT_TRUE(run_test(test1_mesg));
    }
}

TEST(low_level, basic_x_msg_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < 2) {
        EXPECT_TRUE(run_test(test2_mesg));
    }
}

TEST(low_level, basic_x_shared_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank < 2) {
        EXPECT_TRUE(run_test(test1_shared_mesg));
    }
}

