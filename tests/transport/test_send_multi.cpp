#include <transport_layer/mpi/communicator.hpp>
#include <iostream>
#include <iomanip>

int rank;


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

        comm.send_multi(smsg, dsts, 42);
    } else {
        mpi::message<> rmsg{SIZE, SIZE};
        auto fut = comm.recv(rmsg, 0, 42);
        fut.wait();

        bool ok = true;
        for (int i = 0; i < rmsg.size()/sizeof(int); ++i) {
            //std::cout << rmsg. template at<int>(i*sizeof(int)) << ", ";
            if ( rmsg. template at<int>(i*sizeof(int)) != i )
                ok = false;
        }
        std::cout << "\nResult: " << (ok?"PASSED\n":"FAILED\n");

    }

    int c = 0;
    while (comm.progress()) { c++; };
    if (rank == 0) {
        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();

}