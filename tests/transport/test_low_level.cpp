#include <transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>

#include <gtest/gtest.h>

int rank;

/**
 * Simple Send recv on two ranks. P0 sends a message, P1 receives it and check the content.
 */

void test1() {
    gridtools::ghex::mpi::communicator sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    if ( rank == 0 ) {
        sr.blocking_send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

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

        int j = 1;
        for (auto i : rmsg) {
            EXPECT_EQ(static_cast<int>(i), j);
            ++j;
        }
    }

    EXPECT_FALSE(sr.progress());
}

void test2() {
    gridtools::ghex::mpi::communicator sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg;
    std::vector<unsigned char> rmsg2(10);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(std::move(rmsg2), 0, 1, [ &arrived,&rmsg](int /*src*/, int /* tag */, std::vector<unsigned char>&& x) {
            arrived = true;
            rmsg = std::move(x);
        });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            sr.progress();
         } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int j = 1;
        for (auto i : rmsg) {
            EXPECT_EQ(static_cast<int>(i), j);
            ++j;
        }
    }

    EXPECT_FALSE(sr.progress());

}

void test1_mesg() {
    gridtools::ghex::mpi::communicator sr;

    gridtools::ghex::mpi::message<> smsg{40, 40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::mpi::message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.blocking_send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

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

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }

    EXPECT_FALSE(sr.progress());

}

void test2_mesg() {
    gridtools::ghex::mpi::communicator sr;

    gridtools::ghex::mpi::message<> smsg{40, 40};
    gridtools::ghex::mpi::message<> rmsg2{40, 40};
    gridtools::ghex::mpi::message<> rmsg;

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(std::move(rmsg2), 0, 1, [ &arrived, &rmsg](int /* src */, int /* tag */, gridtools::ghex::mpi::message<>&& x) {
            arrived = true;
            rmsg = std::move(x);
        });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            sr.progress();
         } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }

    EXPECT_FALSE(sr.progress());

    MPI_Barrier(MPI_COMM_WORLD);

}

void test1_shared_mesg() {
    gridtools::ghex::mpi::communicator sr;

    gridtools::ghex::mpi::message<> smsg{40, 40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::mpi::shared_message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.blocking_send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

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

        int* data = rmsg.data<int>();
        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }

    EXPECT_FALSE(sr.progress());
}

void test2_shared_mesg() {
    gridtools::ghex::mpi::communicator sr;

    gridtools::ghex::mpi::message<> smsg{40, 40};

    int * data = smsg.data<int>();

    for (int i = 0; i < 10; ++i) {
        data[i] = i;
    }

    gridtools::ghex::mpi::shared_message<> rmsg2{40, 40};
    gridtools::ghex::mpi::shared_message<> rmsg;

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(std::move(rmsg2), 0, 1, [ &arrived, &rmsg](int, int, gridtools::ghex::mpi::shared_message<>&& x) {
            arrived = true;
            rmsg = std::move(x);
        });

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
            sr.progress();
         } while (!arrived);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        int* data = rmsg.data<int>();

        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(data[i], i);
        }
    }

    EXPECT_FALSE(sr.progress());

}

template <typename Msg>
void print_msg(Msg const msg) {
    std::cout << "Reference count " << msg.use_count() << " (size: " << msg.size() << ")\n";
    int * data = msg.template data<int>();
    for (int i = 0; i < (int)(msg.size()/sizeof(int)); ++i) {
        std::cout << data[i] << ", ";
    }
    std::cout << "\n";
}

TEST(transport, basic) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1();
}

TEST(transport, basic_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test2();
}

TEST(transport, basic_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1_mesg();
}

TEST(transport, basic_msg_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test2_mesg();
}

TEST(transport, basic_shared_msg) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1_shared_mesg();
}

TEST(transport, basic_shared_message_call_back) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test2_shared_mesg();
}
