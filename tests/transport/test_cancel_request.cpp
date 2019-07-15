int rank;

#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>
#include <functional>

const int SIZE = 1<<22;

bool test_simple(mpi::communicator &comm, int rank) {
    if (rank == 0) {


        mpi::shared_message<> smsg{SIZE};
        for (int i = 0; i < SIZE/sizeof(int); ++i) {
            smsg.enqueue(i);
        }

        std::array<int, 3> dsts = {1,2,3};

        comm.send_multi(smsg, dsts, 42+42);
        bool ok = comm.cancel_call_backs();
        return ok;
    } else {
        mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42);

        bool ok = fut.cancel();
        return ok;
    }

}

class call_back {
    int & m_value;
    mpi::communicator& m_comm;
    mpi::message<>& m_msg;

public:
    call_back(int& a, mpi::communicator& c, mpi::message<>& m)
    : m_value(a)
    , m_comm{c}
    , m_msg{m}
    { }

    void operator()(int, int) {
        m_value = m_msg.at<int>(0);
        m_comm.recv(m_msg, 0, 42+m_value+1, *this);
    }
};

bool test_send_10(mpi::communicator &comm, int rank) {
    while (comm.progress()) {}
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank == 0) {
        mpi::shared_message<> smsg{sizeof(int), sizeof(int)};
        for (int i = 0; i < 10; ++i) {
            int v = i+666;
            smsg.at<int>(0) = v;

            std::array<int, 3> dsts = {1,2,3};

            comm.send_multi(smsg, dsts, 42+v);
        }
        while (comm.progress()) {}
        return true;
    } else {
        int value = -11111111;

        mpi::message<> rmsg{sizeof(int), sizeof(int)};

        comm.recv(rmsg, 0, 42+666, call_back{value, comm, rmsg});

        while (value < 9+666) {
            comm.progress();
        }

        bool ok = comm.cancel_call_backs();

        return ok;

    }
    return false;
}

#define TEST(x) std::cout << #x << ": " << std::boolalpha << x << "\n";

int main(int argc, char** argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi::communicator comm;

    TEST(test_simple(comm, rank));

    TEST(test_send_10(comm, rank));

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
