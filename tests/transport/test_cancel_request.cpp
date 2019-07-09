int rank;

#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>


const int SIZE = 4000000;
int main(int argc, char** argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    mpi::communicator comm;

    if (rank == 0) {


        mpi::shared_message<> smsg{SIZE};
        for (int i = 0; i < SIZE/sizeof(int); ++i) {
            smsg.enqueue(i);
        }

        std::array<int, 3> dsts = {1,2,3};

        comm.send_multi(smsg, dsts, 42+42);
        bool ok = comm.cancel_call_backs();
        std::cout  << "\nSend Result: " << (ok?"PASSED\n":"FAILED\n");
    } else {
        mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42);

        bool ok = fut.cancel();
        std::cout  << "\nRecv Result: " << (ok?"PASSED\n":"FAILED\n");

    }

    std::cout << rank << ": Waiting on barrier\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}
