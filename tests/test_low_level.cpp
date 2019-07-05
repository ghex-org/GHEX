#include <transport_layer/mpi/communicator.hpp>
#include <vector>

int rank;

void test1() {
    mpi::communicator<int, int> sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else {
        auto fut = sr.recv(rmsg, 0, 1);

        int c = 0;
        do {
            c++;
         } while (fut.ready());

        std::cout << "\n" << c << "\n" << std::endl;
        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\ndone\n";
    }
}

void test2() {
    mpi::communicator<int, int> sr;

    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else {
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });

        int c = 0;
        do {
            c++;
            sr.progress();
         } while (!arrived);

        std::cout << "\n" << c << "\n" << std::endl;
        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\ndone\n";
    }
}

void test1_mesg() {
    mpi::communicator<int, int> sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else {
        auto fut = sr.recv(rmsg, 0, 1);

        int c = 0;
        do {
            c++;
         } while (fut.ready());

        std::cout << "\n" << c << "\n" << std::endl;
        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\nPrint as int:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << rmsg.at<int>(i*sizeof(int)) << ", ";
        }

        std::cout << "\ndone\n";
    }
}

void test2_mesg() {
    mpi::communicator<int, int> sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        fut.wait();
    } else {
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });

        int c = 0;
        do {
            c++;
            sr.progress();
         } while (!arrived);

        std::cout << "\n" << c << "\n" << std::endl;
        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\nPrint as int:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << rmsg.at<int>(i*sizeof(int)) << ", ";
        }
        std::cout << "\ndone\n";
    }
}

int main(int argc, char** argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    test1();

    MPI_Barrier(MPI_COMM_WORLD);

    test2();

    MPI_Barrier(MPI_COMM_WORLD);

    test1_mesg();

    MPI_Barrier(MPI_COMM_WORLD);

    test2_mesg();

    MPI_Finalize();
}
