#include <transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>

#include <gtest/gtest.h>

int rank;

/**
 * Simple Send recv on two ranks. P0 sends a message, P1 receives it and check the content.
 */

void test1() {
    mpi::communicator sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (fut.ready());

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
    mpi::communicator sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(rmsg, 0, 1, [ &arrived](int /*src*/, int /* tag */) {
            arrived = true;
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
    mpi::communicator sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(rmsg.at<int>(i*sizeof(int)), i);
        }
    }

    EXPECT_FALSE(sr.progress());

}

void test2_mesg() {
    mpi::communicator sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(rmsg, 0, 1, [ &arrived](int /* src */, int /* tag */) {
            arrived = true;
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

        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(rmsg.at<int>(i*sizeof(int)), i);
        }
    }

    EXPECT_FALSE(sr.progress());

}

void test1_shared_mesg() {
    mpi::communicator sr;

    mpi::shared_message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::shared_message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else if (rank == 1) {
        auto fut = sr.recv(rmsg, 0, 1);

#ifdef GHEX_TEST_COUNT_ITERATIONS
        int c = 0;
#endif
        do {
#ifdef GHEX_TEST_COUNT_ITERATIONS
            c++;
#endif
         } while (fut.ready());

#ifdef GHEX_TEST_COUNT_ITERATIONS
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
#endif

        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(rmsg.at<int>(i*sizeof(int)), i);
        }
    }

    EXPECT_FALSE(sr.progress());
}

void test2_shared_mesg() {
    mpi::communicator sr;

    mpi::shared_message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::shared_message<> rmsg{40, 40};

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else if (rank == 1) {
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
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

        for (int i = 0; i < 10; ++i) {
            EXPECT_EQ(rmsg.at<int>(i*sizeof(int)), i);
        }
    }

    EXPECT_FALSE(sr.progress());

}

template <typename Msg>
void print_msg(Msg const msg) {
    std::cout << "Reference count " << msg.use_count() << " (size: " << msg.size() << ")\n";
    for (int i = 0; i < (int)(msg.size()/sizeof(int)); ++i) {
        std::cout << msg. template at<int>(i*sizeof(int)) << ", ";
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
