#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>
#include <functional>


#include <gtest/gtest.h>

int rank;
const unsigned int SIZE = 1<<12;

bool test_simple(gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;

    gridtools::ghex::tl::callback_communicator<comm_type,allocator_type> cb_comm(comm);

    if (rank == 0) {
        smsg_type smsg{SIZE};

        int* data = smsg.data<int>();

        for (unsigned int i = 0; i < SIZE/sizeof(int); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        cb_comm.send_multi(smsg, dsts, 42+42); // ~wrong tag to then cancel the calls
        bool ok = cb_comm.cancel();
        MPI_Barrier(comm);
        return ok;
    } else {
        gridtools::ghex::tl::message_buffer<> rmsg{SIZE};
        auto fut = comm.recv(rmsg, 0, 42); // ~wrong tag to then cancel the calls

        bool ok = fut.cancel();

        MPI_Barrier(comm);
        // cleanup msg
        for (int i=0; i<100; ++i)
            cb_comm.progress([](const smsg_type& m, int src,int tag){ 
                std::cout << "received unexpected message from rank " << src << " and tag " << tag 
                << " with size = " << m.size() << std::endl;});

        return ok;
    }

}

bool test_single(gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;

    gridtools::ghex::tl::callback_communicator<comm_type,allocator_type> cb_comm(comm);

    if (rank == 0) {
        smsg_type smsg{SIZE};

        std::array<int, 3> dsts = {1,2,3};

        for (int dst : dsts) {
            cb_comm.send(smsg, dst, 45, [](const smsg_type&, int,int) {} );
        }

        bool ok = true;

        for (auto dst : dsts) {
            if (auto o = cb_comm.detach_send(dst, 45))
                if (!o->first.ready())
                    ok &= o->first.cancel();
        }

        while (cb_comm.progress()) {}

        MPI_Barrier(comm);
        return ok;

    } else {
        bool ok = true;
        smsg_type rmsg{SIZE};

        cb_comm.recv(rmsg, 0, 43, [](const smsg_type&, int, int) {  }); 

        // progress should not be empty
        ok = ok && cb_comm.progress();

        // detach all registered recvs/callbacks and cancel recv operation
        if (auto o = cb_comm.detach_recv(0,43))
        {
            ok = ok && o->first.cancel();
            std::cout << "detached msg size = " << o->second.size() << std::endl;
        }

        // progress shoud be empty now
        ok = ok && !cb_comm.progress();
        while (cb_comm.progress()) {}

        MPI_Barrier(comm);

        // try to cleanup lingering messages
        for (int i=0; i<100; ++i)
            cb_comm.progress([](const smsg_type& m,int src,int tag){ 
                std::cout << "received unexpected message from rank " << src << " and tag " << tag 
                << " with size = " << m.size() << std::endl;});

        return ok;
    }

}


template<typename CBComm>
class call_back {
    int & m_value;
    CBComm& m_cb_comm;

public:
    call_back(int& a, CBComm& p)
    : m_value(a)
    , m_cb_comm{p}
    { }

    void operator()(gridtools::ghex::tl::shared_message_buffer<> m, int, int) 
    {
        m_value = m.data<int>()[0];
        //gridtools::ghex::tl::shared_message_buffer<> m2{m};
        m_cb_comm.recv(m, 0, 42+m_value+1, *this);
    }
};

bool test_send_10(gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::tl::shared_message_buffer<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;
    using cb_comm_type   = gridtools::ghex::tl::callback_communicator<comm_type,allocator_type>;
   
    cb_comm_type cb_comm(comm);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        smsg_type smsg{sizeof(int)};
        for (int i = 0; i < 10; ++i) {
            int v = i;
            smsg.data<int>()[0] = v;

            std::array<int, 3> dsts = {1,2,3};

            cb_comm.send_multi(smsg, dsts, 42+v);
        }
        while (cb_comm.progress()) {}
        return true;
    } else {
        int value = -11111111;

        smsg_type rmsg{sizeof(int)};

        cb_comm.recv(rmsg, 0, 42, call_back<cb_comm_type>{value, cb_comm});

        while (value < 9) {
            cb_comm.progress();
        }

        bool ok = cb_comm.cancel();

        return ok;

    }
    return false;
}

TEST(transport, check_mpi_ranks_eq_4) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    EXPECT_EQ(size, 4);
}

TEST(transport, cancel_requests_reposting) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> comm;

    EXPECT_TRUE(test_send_10(comm, rank));

}

TEST(transport, cancel_requests_simple) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> comm;

    EXPECT_TRUE(test_simple(comm, rank));

}

TEST(transport, cancel_single_request) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> comm;

    EXPECT_TRUE(test_single(comm, rank));
}

