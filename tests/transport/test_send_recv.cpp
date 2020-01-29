#include <iostream>
#include <iomanip>
#include <ghex/threads/none/primitives.hpp>
#include <ghex/common/timer.hpp>
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
using communicator_type = typename context_type::communicator_type;

#define NITERS 100000

TEST(transport, ring_send_recv_ft)
{
    gridtools::ghex::timer timer;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    int size = context.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    std::vector<int> smsg(1), rmsg(1);

    timer.tic();
    for(int i=0; i<NITERS; i++){
        communicator_type::future<void> sreq, rreq;
        smsg[0] = rank;
        rreq = comm.recv(rmsg, rpeer_rank, 1);
        sreq = comm.send(smsg, speer_rank, 1);
        while(!(rreq.ready() && sreq.ready()));
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
    
    EXPECT_TRUE(rmsg[0] == rpeer_rank);
}

TEST(transport, ring_send_recv_cb)
{
    gridtools::ghex::timer timer;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    int size = context.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    std::vector<int> smsg(1), rmsg(1);

    timer.tic();
    for(int i=0; i<NITERS; i++){

        volatile int received = 0;
        volatile int sent = 0;

        auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
        auto recv_callback = [&](communicator_type::message_type, int, int) {received++;};    

        smsg[0] = rank;
        comm.recv(rmsg, rpeer_rank, 1, recv_callback);
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received!=1 || sent!=1) comm.progress();
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
    
    EXPECT_TRUE(rmsg[0] == rpeer_rank);
}


TEST(transport, ring_send_recv_cb_disown)
{
    gridtools::ghex::timer timer;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    int size = context.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();
    for(int i=0; i<NITERS; i++){

        std::vector<int> smsg(1), rmsg(1);
        volatile int received = 0;
        volatile int sent = 0;

        auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
        auto recv_callback = [&](communicator_type::message_type rmsg, int, int) {
            int *ptr = reinterpret_cast<int*>(rmsg.data());
            EXPECT_TRUE(ptr[0] == rpeer_rank);
            received++;
        };
        smsg[0] = rank;
        comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
        comm.send(std::move(smsg), speer_rank, 1, send_callback);
        while(received!=1 || sent!=1) comm.progress();        
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}


TEST(transport, ring_send_recv_cb_resubmit)
{
    gridtools::ghex::timer timer;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    int size = context.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();
    std::vector<int> smsg(1), rmsg(1);
    communicator_type::request_cb_type rreq;
    
    volatile int received = 0;
    volatile int sent = 0;
    auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
    std::function<void(communicator_type::message_type rmsg, int, int)> recv_callback;
    recv_callback = [&](communicator_type::message_type rmsg, int, int) {
        int *ptr = reinterpret_cast<int*>(rmsg.data());
        EXPECT_TRUE(ptr[0] == rpeer_rank);
        received++;
        rreq = comm.recv(rmsg, rpeer_rank, 1, recv_callback);
    };

    rreq = comm.recv(rmsg, rpeer_rank, 1, recv_callback);
    smsg[0] = rank;
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received<i || sent<i) comm.progress();
    }
    rreq.cancel();
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}


TEST(transport, ring_send_recv_cb_resubmit_disown)
{
    gridtools::ghex::timer timer;
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    auto token = context.get_token();
    auto comm = context.get_communicator(token);
    int rank = context.rank();
    int size = context.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();
    std::vector<int> smsg(1), rmsg(1);
    communicator_type::request_cb_type rreq;

    volatile int received = 0;
    volatile int sent = 0;
    auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
    std::function<void(communicator_type::message_type rmsg, int, int)> recv_callback;
    recv_callback = [&](communicator_type::message_type rmsg, int, int) {
        int *ptr = reinterpret_cast<int*>(rmsg.data());
        EXPECT_TRUE(ptr[0] == rpeer_rank);
        received++;
        rreq = comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
    };

    comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
    smsg[0] = rank;
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received<i || sent<i) comm.progress();
    }
    rreq.cancel();
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}
