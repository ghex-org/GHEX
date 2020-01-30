#include <iostream>
#include <iomanip>
#include <ghex/threads/none/primitives.hpp>
#include <ghex/transport_layer/message_buffer.hpp>
#include <ghex/transport_layer/shared_message_buffer.hpp>
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


template<typename MsgType, typename CommType>
auto test_ring_send_recv_ft(CommType& comm, MsgType& rmsg, MsgType& smsg) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    timer.tic();
    for(int i=0; i<NITERS; i++){

        auto rreq = comm.recv(rmsg, rpeer_rank, 1);
        auto sreq = comm.send(smsg, speer_rank, 1);
        while(!(rreq.ready() && sreq.ready()));

        data_ptr = reinterpret_cast<int*>(rmsg.data());
        EXPECT_TRUE(*data_ptr == rpeer_rank);
    }
    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

template<typename MsgType, typename CommType>
auto test_ring_send_recv_ft(CommType& comm) 
{
    MsgType smsg(4), rmsg(4);
    test_ring_send_recv_ft(comm, rmsg, smsg);
}

TEST(transport, ring_send_recv_ft)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_ft<std::vector<unsigned char>>(comm);
    test_ring_send_recv_ft<gridtools::ghex::tl::message_buffer<>>(comm);
    test_ring_send_recv_ft<gridtools::ghex::tl::shared_message_buffer<>>(comm);
    
    {
        gridtools::ghex::tl::message_buffer<> smsg(4), rmsg(4);
        communicator_type::message_type asmsg{std::move(smsg)}, armsg{std::move(rmsg)};
        test_ring_send_recv_ft(comm, std::move(asmsg), std::move(armsg));
    }
}



template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb(CommType& comm, MsgType &&rmsg, MsgType &&smsg) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    timer.tic();
    volatile int received = 0;
    volatile int sent = 0;
    for(int i=0; i<NITERS; i++){

        auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
        auto recv_callback = [&](communicator_type::message_type, int, int) {received++;};    

        comm.recv(rmsg, rpeer_rank, 1, recv_callback);
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received<=i || sent<=i) comm.progress();

        data_ptr = reinterpret_cast<int*>(rmsg.data());
        EXPECT_TRUE(*data_ptr == rpeer_rank);
    }

    EXPECT_TRUE(received==NITERS && sent==NITERS);

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb(CommType& comm) 
{
    MsgType smsg(4), rmsg(4);
    test_ring_send_recv_cb(comm, std::move(rmsg), std::move(smsg));
}

TEST(transport, ring_send_recv_cb)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb<std::vector<unsigned char>>(comm);
    test_ring_send_recv_cb<gridtools::ghex::tl::message_buffer<>>(comm);
    test_ring_send_recv_cb<gridtools::ghex::tl::shared_message_buffer<>>(comm);
    
    {
        gridtools::ghex::tl::message_buffer<> smsg(4), rmsg(4);
        communicator_type::message_type asmsg{std::move(smsg)}, armsg{std::move(rmsg)};
        test_ring_send_recv_cb(comm, std::move(armsg), std::move(asmsg));
    }
}



template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_disown(CommType& comm, MsgType &&rmsg, MsgType &&smsg) 
{
    using MsgType_v = typename std::remove_reference_t<MsgType>;

    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();
    volatile int received = 0;
    volatile int sent = 0;

    auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
    auto recv_callback = [&](communicator_type::message_type mrmsg, int, int) 
        {
            received++;
            int *data_ptr = reinterpret_cast<int*>(mrmsg.data());
            EXPECT_TRUE(*data_ptr == rpeer_rank);
        };

    for(int i=0; i<NITERS; i++){

        MsgType_v mrmsg(rmsg.size()), msmsg(smsg.size());
        data_ptr = reinterpret_cast<int*>(msmsg.data());
        *data_ptr = rank;

        comm.recv(std::move(mrmsg), rpeer_rank, 1, recv_callback);
        comm.send(std::move(msmsg), speer_rank, 1, send_callback);
        while(received<=i || sent<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS && sent==NITERS);

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_disown(CommType& comm) 
{
    MsgType smsg(4), rmsg(4);
    test_ring_send_recv_cb_disown(comm, std::move(rmsg), std::move(smsg));
}

TEST(transport, ring_send_recv_cb_disown)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_disown<std::vector<unsigned char>>(comm);
    test_ring_send_recv_cb_disown<gridtools::ghex::tl::message_buffer<>>(comm);
    test_ring_send_recv_cb_disown<gridtools::ghex::tl::shared_message_buffer<>>(comm);
}



template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_resubmit(CommType& comm, MsgType &&rmsg, MsgType &&smsg) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();    
    communicator_type::request_cb_type rreq;
    volatile int received = 0;
    volatile int sent = 0;
    auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};

    std::function<void(communicator_type::message_type rmsg, int, int)> recv_callback;
    recv_callback = [&](communicator_type::message_type rmsg, int, int) 
        {
            received++;
            int *data_ptr = reinterpret_cast<int*>(rmsg.data());
            EXPECT_TRUE(*data_ptr == rpeer_rank);
            rreq = comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
        };

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    rreq = comm.recv(rmsg, rpeer_rank, 1, recv_callback);
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received<=i || sent<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS && sent==NITERS);
    EXPECT_TRUE(rreq.cancel());

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_resubmit(CommType& comm) 
{
    MsgType smsg(4), rmsg(4);
    test_ring_send_recv_cb_resubmit(comm, std::move(rmsg), std::move(smsg));
}

TEST(transport, ring_send_recv_cb_resubmit)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_resubmit<std::vector<unsigned char>>(comm);
    test_ring_send_recv_cb_resubmit<gridtools::ghex::tl::message_buffer<>>(comm);
    test_ring_send_recv_cb_resubmit<gridtools::ghex::tl::shared_message_buffer<>>(comm);
    
    {
        gridtools::ghex::tl::message_buffer<> smsg(4), rmsg(4);
        communicator_type::message_type asmsg{std::move(smsg)}, armsg{std::move(rmsg)};
        test_ring_send_recv_cb_resubmit(comm, std::move(asmsg), std::move(armsg));
    }
}



template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_resubmit_disown(CommType& comm, MsgType &&rmsg, MsgType &&smsg) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    timer.tic();    
    communicator_type::request_cb_type rreq;
    volatile int received = 0;
    volatile int sent = 0;
    auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};

    std::function<void(communicator_type::message_type rmsg, int, int)> recv_callback;
    recv_callback = [&](communicator_type::message_type rmsg, int, int) 
        {
            received++;
            int *data_ptr = reinterpret_cast<int*>(rmsg.data());
            EXPECT_TRUE(*data_ptr == rpeer_rank);
            rreq = comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
        };

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    rreq = comm.recv(std::move(rmsg), rpeer_rank, 1, recv_callback);
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1, send_callback);
        while(received<=i || sent<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS && sent==NITERS);
    EXPECT_TRUE(rreq.cancel());

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

template<typename MsgType, typename CommType>
auto test_ring_send_recv_cb_resubmit_disown(CommType& comm) 
{
    MsgType smsg(4), rmsg(4);
    test_ring_send_recv_cb_resubmit_disown(comm, std::move(rmsg), std::move(smsg));
}

TEST(transport, ring_send_recv_cb_resubmit_disown)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_resubmit_disown<std::vector<unsigned char>>(comm);
    test_ring_send_recv_cb_resubmit_disown<gridtools::ghex::tl::message_buffer<>>(comm);
    test_ring_send_recv_cb_resubmit_disown<gridtools::ghex::tl::shared_message_buffer<>>(comm);
    
    {
        gridtools::ghex::tl::message_buffer<> smsg(4), rmsg(4);
        communicator_type::message_type asmsg{std::move(smsg)}, armsg{std::move(rmsg)};
        test_ring_send_recv_cb_resubmit_disown(comm, std::move(asmsg), std::move(armsg));
    }
}
