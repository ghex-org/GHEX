#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/context.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

template<typename Comm, typename Alloc>
using callback_comm_t = gridtools::ghex::tl::callback_communicator<Comm,Alloc>;
//using callback_comm_t = gridtools::ghex::tl::callback_communicator_ts<Comm,Alloc>;

using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::atomic::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

const int SIZE = 4000000;
int mpi_rank;

//#define GHEX_TEST_COUNT_ITERATIONS

TEST(transport, send_multi) {

    {
        int size;
        MPI_Comm_size(MPI_COMM_WORLD, &size);
        EXPECT_EQ(size, 4);
    }

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    mpi_rank = context.world().rank();
    context.barrier(token);

    auto comm = context.get_communicator(token);
    using comm_type = std::remove_reference_t<decltype(comm)>;

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;

    callback_comm_t<comm_type,allocator_type> cb_comm(comm);

    if (mpi_rank == 0) {

        smsg_type smsg{SIZE};

        int * data = smsg.data<int>();

        for (int i = 0; i < SIZE/(int)sizeof(int); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        cb_comm.send_multi(smsg, dsts, 42);

#ifdef GHEX_TEST_COUNT_ITERATIONS
    int c = 0;
#endif
    while (cb_comm.progress()) {
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
        gridtools::ghex::tl::message_buffer<> rmsg{SIZE};
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


    EXPECT_FALSE(cb_comm.progress());
}

