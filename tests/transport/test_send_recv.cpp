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
using msg_type = typename communicator_type::message_type;

#define NITERS 100000

template<typename Message>
struct message_factory {
    static Message make(std::size_t size) { return Message(size); }
};

template<>
struct message_factory<msg_type> {
    static msg_type make(std::size_t size) { return {std::vector<unsigned char>(size)}; }
};

template<typename Factory, typename CommType>
auto test_ring_send_recv_ft(CommType& comm, std::size_t buffer_size) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    auto smsg = Factory::make(buffer_size);
    auto rmsg = Factory::make(buffer_size);

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    comm.barrier();
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

TEST(transport, ring_send_recv_ft)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_ft< message_factory<std::vector<unsigned char>> >(comm, sizeof(int));
    test_ring_send_recv_ft< message_factory<gridtools::ghex::tl::message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_ft< message_factory<gridtools::ghex::tl::shared_message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_ft< message_factory<msg_type> >(comm, sizeof(int));
}



template<typename Factory, typename CommType>
auto test_ring_send_recv_cb(CommType& comm, std::size_t buffer_size) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;
    
    auto smsg = Factory::make(buffer_size);
    auto rmsg = Factory::make(buffer_size);

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;

    comm.barrier();
    //comm.progress();
    timer.tic();
    volatile int received = 0;
    volatile int sent = 0;
    for(int i=0; i<NITERS; i++){

        auto send_callback = [&](communicator_type::message_type, int, int) {sent++;};
        auto recv_callback = [&](communicator_type::message_type, int, int) {received++;};    

        comm.recv(rmsg, rpeer_rank, 1, recv_callback);
        comm.send(smsg, speer_rank, 1, send_callback);
        auto status = comm.progress();
        while(received<=i || sent<=i) { status += comm.progress(); }
        EXPECT_EQ(status.num_sends(), 1);
        EXPECT_EQ(status.num_recvs(), 1);

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

TEST(transport, ring_send_recv_cb)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb< message_factory<std::vector<unsigned char>> >(comm, sizeof(int));
    test_ring_send_recv_cb< message_factory<gridtools::ghex::tl::message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb< message_factory<gridtools::ghex::tl::shared_message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb< message_factory<msg_type> >(comm, sizeof(int));
}

template<typename Factory, typename CommType>
auto test_ring_send_recv_cb_disown(CommType& comm, std::size_t buffer_size) 
{
    gridtools::ghex::timer timer;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank-1)%size;
    if(rpeer_rank<0) rpeer_rank = size-1;

    comm.barrier();
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

        auto smsg = Factory::make(buffer_size);
        *reinterpret_cast<int*>(smsg.data()) = rank;

        comm.recv(Factory::make(buffer_size), rpeer_rank, 1, recv_callback);
        comm.send(std::move(smsg), speer_rank, 1, send_callback);
        while(received<=i || sent<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS && sent==NITERS);

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

TEST(transport, ring_send_recv_cb_disown)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_disown< message_factory<std::vector<unsigned char>> >(comm, sizeof(int));
    test_ring_send_recv_cb_disown< message_factory<gridtools::ghex::tl::message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_disown< message_factory<gridtools::ghex::tl::shared_message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_disown< message_factory<msg_type> >(comm, sizeof(int));
}

struct recursive_functor {
    communicator_type& comm;
    int rank;
    int tag;
    typename communicator_type::request_cb& rreq;
    std::function<void(msg_type&, int, int)> func;
    // we need a counter here so we can handle immediate callbacks
    int counter = 0;

    void operator()(msg_type m, int r, int t) {
        func(m,r,t);
        counter = 0;
        auto rr = comm.recv(std::move(m), rank, tag, *this);
        if (counter == 0) {
            ++counter;
            rreq = std::move(rr);
        }
    }
};

template<typename Factory, typename CommType>
auto test_ring_send_recv_cb_resubmit(CommType& comm, std::size_t buffer_size) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank+size-1)%size;
    
    auto smsg = Factory::make(buffer_size);
    auto rmsg = Factory::make(buffer_size);

    comm.barrier();
    timer.tic();    
    volatile int received = 0;

    typename communicator_type::request_cb rreq;
    auto recv_callback = [&received,rpeer_rank](communicator_type::message_type& rmsg, int, int) 
        {
            received++;
            int *data_ptr = reinterpret_cast<int*>(rmsg.data());
            EXPECT_TRUE(*data_ptr == rpeer_rank);
            *data_ptr = -1;
        };
    auto recursive_recv_callback = recursive_functor{comm,rpeer_rank,1,rreq,recv_callback};

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;


    rreq = comm.recv(rmsg, rpeer_rank, 1, recursive_recv_callback);
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1).wait();
        while(received<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS);
    EXPECT_TRUE(rreq.cancel());

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

TEST(transport, ring_send_recv_cb_resubmit)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_resubmit< message_factory<std::vector<unsigned char>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit< message_factory<gridtools::ghex::tl::message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit< message_factory<gridtools::ghex::tl::shared_message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit< message_factory<msg_type> >(comm, sizeof(int));
}

template<typename Factory, typename CommType>
auto test_ring_send_recv_cb_resubmit_disown(CommType& comm, std::size_t buffer_size) 
{
    gridtools::ghex::timer timer;
    int *data_ptr;
    int rank = comm.rank();
    int size = comm.size();
    int speer_rank = (rank+1)%size;
    int rpeer_rank = (rank+size-1)%size;
    
    auto smsg = Factory::make(buffer_size);

    comm.barrier();
    timer.tic(); 
    volatile int received = 0;

    typename communicator_type::request_cb rreq;
    auto recv_callback = [&received,rpeer_rank](communicator_type::message_type& rmsg, int, int) 
        {
            received++;
            int *data_ptr = reinterpret_cast<int*>(rmsg.data());
            EXPECT_TRUE(*data_ptr == rpeer_rank);
            *data_ptr = -1;
        };
    auto recursive_recv_callback = recursive_functor{comm,rpeer_rank,1,rreq,recv_callback};

    data_ptr = reinterpret_cast<int*>(smsg.data());
    *data_ptr = rank;


    rreq = comm.recv(Factory::make(buffer_size), rpeer_rank, 1, recursive_recv_callback);
    for(int i=0; i<NITERS; i++){
        comm.send(smsg, speer_rank, 1).wait();
        while(received<=i) comm.progress();
    }

    EXPECT_TRUE(received==NITERS);
    EXPECT_TRUE(rreq.cancel());

    const auto t = timer.stoc();
    if(rank==0)
    {
        std::cout << "time:       " << t/1000000 << "s\n";
    }
}

TEST(transport, ring_send_recv_cb_resubmit_disown)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto token = context.get_token();
    auto comm = context.get_communicator(token);

    test_ring_send_recv_cb_resubmit_disown< message_factory<std::vector<unsigned char>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit_disown< message_factory<gridtools::ghex::tl::message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit_disown< message_factory<gridtools::ghex::tl::shared_message_buffer<>> >(comm, sizeof(int));
    test_ring_send_recv_cb_resubmit_disown< message_factory<msg_type> >(comm, sizeof(int));
}

