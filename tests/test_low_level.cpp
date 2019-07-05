#include <transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>

int rank;

void test1() {
    mpi::communicator sr;

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

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";

        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\ndone\n";
    }
}

void test2() {
    mpi::communicator sr;

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

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";

        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\ndone\n";
    }
}

void test1_mesg() {
    mpi::communicator sr;

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

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";

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
    mpi::communicator sr;

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

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";

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

auto test1_shared_mesg() {
    mpi::communicator sr;

    mpi::shared_message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::shared_message<> rmsg{40, 40};

    if ( rank == 0 ) {
        sr.send(smsg, 1, 1);
    } else {
        auto fut = sr.recv(rmsg, 0, 1);

        int c = 0;
        do {
            c++;
         } while (fut.ready());

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";

        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\nPrint as int:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << rmsg.at<int>(i*sizeof(int)) << ", ";
        }

        std::cout << "\ndone\n";
    }

    return rmsg;
}

auto test2_shared_mesg() {
    mpi::communicator sr;

    mpi::shared_message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::shared_message<> rmsg{40, 40};

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

        std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
        std::cout << "***********\n";
        for (auto i : rmsg) {
            std::cout << static_cast<int>(i) << ", ";
        }
        std::cout << "\nPrint as int:\n";
        for (int i = 0; i < 10; ++i) {
            std::cout << rmsg.at<int>(i*sizeof(int)) << ", ";
        }
        std::cout << "\ndone\n";
    }

    return rmsg;
}

template <typename Msg>
void print_msg(Msg const msg) {
    std::cout << "Reference count " << msg.use_count() << " (size: " << msg.size() << ")\n";
    for (int i = 0; i < msg.size()/sizeof(int); ++i) {
        std::cout << msg. template at<int>(i*sizeof(int)) << ", ";
    }
    std::cout << "\n";
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

    MPI_Barrier(MPI_COMM_WORLD);

    auto msg1 = test1_shared_mesg();

    std::cout << "\nPrint as int from main:\n";
    if (rank==1)
        print_msg(msg1);

    MPI_Barrier(MPI_COMM_WORLD);

    auto msg2 = test2_shared_mesg();

    if (rank==1)
        print_msg(msg2);

    MPI_Finalize();
}
