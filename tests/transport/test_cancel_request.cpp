#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>
#include <functional>


#include <gtest/gtest.h>

int rank;
const unsigned int SIZE = 1<<12;

bool test_simple(gridtools::ghex::mpi::communicator &comm, int rank) {

    if (rank == 0) {
        gridtools::ghex::mpi::shared_message<> smsg{SIZE, SIZE};

        int* data = smsg.data<int>();

        for (unsigned int i = 0; i < SIZE/sizeof(int); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        comm.send_multi(smsg, dsts, 42+42); // ~wrong tag to then cancel the calls
        bool ok = comm.cancel_callbacks();
        return ok;
    } else {
        gridtools::ghex::mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42); // ~wrong tag to then cancel the calls

        bool ok = fut.cancel();

        return ok;
    }

}

bool test_single(gridtools::ghex::mpi::communicator &comm, int rank) {

    if (rank == 0) {
        gridtools::ghex::mpi::shared_message<> smsg{SIZE, SIZE};

        std::array<int, 3> dsts = {1,2,3};
        std::array<gridtools::ghex::mpi::communicator::request_type, 3> reqs;

        for (int dst : dsts) {
            comm.send(smsg, dst, 45, [smsg](int,int) {} );
        }

        bool ok = true;

        for (auto dst : dsts) {
            if (auto fut = comm.detach_send(dst, 45))
                if (!fut->ready())
                    ok &= fut->cancel();
        }

        while (comm.progress()) {}

        return ok;

    } else {
        //comm.recv(gridtools::ghex::mpi::message<>{SIZE, SIZE}, 0, 43, [](int, int, gridtools::ghex::mpi::message<>&&) {  }); // unmatching tag
        comm.recv(0, 43, [](int, int, gridtools::ghex::mpi::message<>&) { }, SIZE, SIZE); // unmatching tag
        bool ok = true;
        if (auto fut = comm.detach_recv(0,43))
            ok = fut->cancel();

        while (comm.progress()) {}

        return ok;
    }

}


class call_back {
    int & m_value;
    gridtools::ghex::mpi::communicator& m_comm;

public:
    call_back(int& a, gridtools::ghex::mpi::communicator& c)
    : m_value(a)
    , m_comm{c}
    { }

    void operator()(int, int, gridtools::ghex::mpi::message<>& m) {
        m_value = m.data<int>()[0];
        m_comm.recv(std::move(m), 0, 42+m_value+1, *this);
    }
};

bool test_send_10(gridtools::ghex::mpi::communicator &comm, int rank) {
    while (comm.progress()) {}
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        gridtools::ghex::mpi::shared_message<> smsg{sizeof(int), sizeof(int)};
        for (int i = 0; i < 10; ++i) {
            int v = i;
            smsg.data<int>()[0] = v;

            std::array<int, 3> dsts = {1,2,3};

            comm.send_multi(smsg, dsts, 42+v);
        }
        while (comm.progress()) {}
        return true;
    } else {
        int value = -11111111;

        //gridtools::ghex::mpi::message<> rmsg{sizeof(int), sizeof(int)};

        comm.recv(gridtools::ghex::mpi::message<>{sizeof(int),sizeof(int)}, 0, 42, call_back{value, comm});

        while (value < 9) {
            comm.progress();
        }

        bool ok = comm.cancel_callbacks();

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
