#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>
#include "../gtest_main_boost.cpp"

const int SIZE = 4000000;
int mpi_rank;


TEST(transport, send_multi) {

    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        EXPECT_EQ(size, 4);
    }


    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    MPI_Barrier(MPI_COMM_WORLD);

    mpi::communicator comm;

    if (mpi_rank == 0) {


        mpi::shared_message<> smsg{SIZE};
        for (int i = 0; i < SIZE/(int)sizeof(int); ++i) {
            smsg.enqueue(i);
        }

        std::array<int, 3> dsts = {1,2,3};

        comm.send_multi(smsg, dsts, 42);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    while (comm.progress()) {
#ifdef GHEX_TEST_COUNT_ITERATIONS
        c++;
#endif
    }

    EXPECT_EQ(smsg.use_count(), 1);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    std::cout  << "\n***********\n";
    std::cout  <<   "*" << std::setw(8) << c << " *\n";
    std::cout  << "***********\n";
#endif


    } else {
        mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42);
        fut.wait();

        bool ok = true;
        for (int i = 0; i < (int)rmsg.size()/(int)sizeof(int); ++i) {
            if ( rmsg. template at<int>(i*sizeof(int)) != i )
                ok = false;
        }

        EXPECT_TRUE(ok);
    }


    EXPECT_FALSE(comm.progress());

}
