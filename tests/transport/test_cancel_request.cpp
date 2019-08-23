#include <transport_layer/progress.hpp>
#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>
#include <functional>


#include <gtest/gtest.h>

int rank;
const unsigned int SIZE = 1<<12;

bool test_simple(gridtools::ghex::mpi::communicator &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::mpi::shared_message<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;

    gridtools::ghex::progress<comm_type,allocator_type> progress(comm);

    if (rank == 0) {
        smsg_type smsg{SIZE, SIZE};

        int* data = smsg.data<int>();

        for (unsigned int i = 0; i < SIZE/sizeof(int); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        progress.send_multi(smsg, dsts, 42+42); // ~wrong tag to then cancel the calls
        bool ok = progress.cancel();
        MPI_Barrier(comm);
        return ok;
    } else {
        gridtools::ghex::mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42); // ~wrong tag to then cancel the calls

        bool ok = fut.cancel();

        MPI_Barrier(comm);
        // cleanup msg
        //comm.recv_any([](int, int, gridtools::ghex::mpi::message<>&) { std::cout << "received unexpected msg!" << std::endl; });
        for (int i=0; i<100; ++i)
            comm.irecv_any([](int, int, gridtools::ghex::mpi::message<>&) { std::cout << "received unexpected msg!" << std::endl; });

        return ok;
    }

}

bool test_single(gridtools::ghex::mpi::communicator &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::mpi::shared_message<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;

    gridtools::ghex::progress<comm_type,allocator_type> progress(comm);

    if (rank == 0) {
        smsg_type smsg{SIZE, SIZE};

        std::array<int, 3> dsts = {1,2,3};
        std::array<gridtools::ghex::mpi::communicator::request_type, 3> reqs;

        for (int dst : dsts) {
            progress.send(smsg, dst, 45, [](int,int,const smsg_type&) {} );
        }

        bool ok = true;

        for (auto dst : dsts) {
            if (auto o = progress.detach_send(dst, 45))
                if (!o->first.ready())
                    ok &= o->first.cancel();
        }

        while (progress()) {}

        MPI_Barrier(comm);
        return ok;

    } else {
        bool ok = true;
        smsg_type rmsg{SIZE, SIZE};

        // recv umatching tag
        progress.recv(rmsg, 0, 43, [](int, int, const smsg_type&) {  }); 

        // progress should not be empty
        ok = ok && progress();

        // detach all registered recvs/callbacks and cancel recv operation
        if (auto o = progress.detach_recv(0,43))
        {
            ok = ok && o->first.cancel();
            std::cout << "detached msg size = " << o->second.size() << std::endl;
        }

        // progress shoud be empty now
        ok = ok && !progress();
        while (progress()) {}

        MPI_Barrier(comm);

        // try to cleanup lingering messages
        for (int i=0; i<100; ++i)
            comm.irecv_any([](int, int, gridtools::ghex::mpi::message<>&) { std::cout << "received unexpected msg!" << std::endl; });

        return ok;
    }

}


template<typename Progress>
class call_back {
    int & m_value;
    Progress& m_progress;

public:
    call_back(int& a, Progress& p)
    : m_value(a)
    , m_progress{p}
    { }

    void operator()(int, int, const gridtools::ghex::mpi::shared_message<>& m) 
    {
        m_value = m.data<int>()[0];
        m_progress.recv(m, 0, 42+m_value+1, *this);
    }
};

bool test_send_10(gridtools::ghex::mpi::communicator &comm, int rank) {

    using allocator_type = std::allocator<unsigned char>;
    using smsg_type      = gridtools::ghex::mpi::shared_message<allocator_type>;
    using comm_type      = std::remove_reference_t<decltype(comm)>;
    using progress_type  = gridtools::ghex::progress<comm_type,allocator_type>;
   
    progress_type progress(comm);

    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        smsg_type smsg{sizeof(int), sizeof(int)};
        for (int i = 0; i < 10; ++i) {
            int v = i;
            smsg.data<int>()[0] = v;

            std::array<int, 3> dsts = {1,2,3};

            progress.send_multi(smsg, dsts, 42+v);
        }
        while (progress()) {}
        return true;
    } else {
        int value = -11111111;

        smsg_type rmsg{sizeof(int), sizeof(int)};

        progress.recv(rmsg, 0, 42, call_back<progress_type>{value, progress});

        while (value < 9) {
            progress();
        }

        bool ok = progress.cancel();

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

    gridtools::ghex::mpi::communicator comm;

    EXPECT_TRUE(test_send_10(comm, rank));

}

TEST(transport, cancel_requests_simple) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::ghex::mpi::communicator comm;

    EXPECT_TRUE(test_simple(comm, rank));

}

TEST(transport, cancel_single_request) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::ghex::mpi::communicator comm;

    EXPECT_TRUE(test_single(comm, rank));
}
