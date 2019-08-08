#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>
#include <functional>


#include <gtest/gtest.h>

int rank;
const int SIZE = 1<<12;

bool test_simple(gridtools::mpi::communicator &comm, int rank) {

    if (rank == 0) {
        gridtools::mpi::shared_message<> smsg{SIZE, SIZE};

        int* data = smsg.data<int>();

        for (int i = 0; i < SIZE/static_cast<int>(sizeof(int)); ++i) {
            data[i] = i;
        }

        std::array<int, 3> dsts = {1,2,3};

        comm.send_multi(smsg, dsts, 42+42); // ~wrong tag to then cancel the calls
        bool ok = comm.cancel_callbacks();
        return ok;
    } else {
        gridtools::mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42); // ~wrong tag to then cancel the calls

        bool ok = fut.cancel();
        return ok;
    }

}

bool test_single(gridtools::mpi::communicator &comm, int rank) {

    if (rank == 0) {
        gridtools::mpi::shared_message<> smsg{SIZE, SIZE};

        std::array<int, 3> dsts = {1,2,3};
        std::array<gridtools::mpi::communicator::request_type, 3> reqs;
        std::array<gridtools::mpi::communicator::send_future, 3> fut;

        int i = 0;
        for (int dst : dsts) {
            reqs[i++] = comm.send(smsg, dst, 45, [smsg](int,int) {} );
        }

        bool ok = true;

        for (auto req : reqs) {
            ok &= comm.cancel_callback(req);
        }


        while (comm.progress()) {}

        return ok;

    } else {
        gridtools::mpi::message<> rmsg{SIZE, SIZE};
        auto req = comm.recv(rmsg, 0, 43, [](int, int) {}); // unmatching tag
        bool ok = comm.cancel_callback(req);

        while (comm.progress()) {}

        return ok;
    }

}


class call_back {
    int & m_value;
    gridtools::mpi::communicator& m_comm;
    gridtools::mpi::message<>& m_msg;

public:
    call_back(int& a, gridtools::mpi::communicator& c, gridtools::mpi::message<>& m)
    : m_value(a)
    , m_comm{c}
    , m_msg{m}
    { }

    void operator()(int, int) {
        m_value = m_msg.data<int>()[0];
        m_comm.recv(m_msg, 0, 42+m_value+1, *this);
    }
};

bool test_send_10(gridtools::mpi::communicator &comm, int rank) {
    while (comm.progress()) {}
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        gridtools::mpi::shared_message<> smsg{sizeof(int), sizeof(int)};
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

        gridtools::mpi::message<> rmsg{sizeof(int), sizeof(int)};

        comm.recv(rmsg, 0, 42, call_back{value, comm, rmsg});

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

    gridtools::mpi::communicator comm;

    EXPECT_TRUE(test_send_10(comm, rank));

}

TEST(transport, cancel_requests_simple) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::mpi::communicator comm;

    EXPECT_TRUE(test_simple(comm, rank));

}

TEST(transport, cancel_single_request) {

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    gridtools::mpi::communicator comm;

    EXPECT_TRUE(test_single(comm, rank));
}
