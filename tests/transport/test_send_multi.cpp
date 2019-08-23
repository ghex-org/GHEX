#include <transport_layer/progress.hpp>
#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

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

    gridtools::ghex::mpi::communicator comm;

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::mpi::shared_message<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;

    gridtools::ghex::progress<comm_type,allocator_type> progress(comm);

    if (mpi_rank == 0) {


        smsg_type smsg{SIZE, SIZE};

        int * data = smsg.data<int>();

        for (int i = 0; i < SIZE/(int)sizeof(int); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        progress.send_multi(smsg, dsts, 42);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    while (progress()) {
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
        gridtools::ghex::mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42);
        fut.wait();

        bool ok = true;
        for (int i = 0; i < (int)rmsg.size()/(int)sizeof(int); ++i) {
            int * data = rmsg.data<int>();
            if ( data[i] != i )
                ok = false;
        }

        EXPECT_TRUE(ok);
    }


    EXPECT_FALSE(progress());

}
