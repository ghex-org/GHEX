#include <transport_layer/mpi/communicator.hpp>
#include <vector>
#include <iomanip>
#include <utility>

int rank;

auto test1() {
    mpi::communicator sr;

    std::vector<unsigned char> smsg = {0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0,8,0,0,0,9,0,0,0};
    std::vector<unsigned char> rmsg(40);

    mpi::communicator::common_future fut;

    if ( rank == 0 ) {
        sr.send_safe(smsg, 1, 1);
        fut = sr.recv(rmsg, 1, 2);
    } else {
        sr.send_safe(smsg, 0, 2);
        fut = sr.recv(rmsg, 0, 1);
    }

    int c = 0;
    do {
        c++;
     } while (fut.ready());

    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

    return rmsg;
}

auto test2() {
    mpi::communicator sr;

    std::vector<unsigned char> smsg = {0,0,0,0,1,0,0,0,2,0,0,0,3,0,0,0,4,0,0,0,5,0,0,0,6,0,0,0,7,0,0,0,8,0,0,0,9,0,0,0};
    std::vector<unsigned char> rmsg(40);

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        sr.recv(rmsg, 1, 2, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    } else {
        auto fut = sr.send(smsg, 0, 2);
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    }

    int c = 0;
    do {
        c++;
        sr.progress();
     } while (!arrived);

    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

    return rmsg;
}

auto test1_mesg() {
    mpi::communicator sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    mpi::communicator::common_future fut;

    if ( rank == 0 ) {
        sr.send_safe(smsg, 1, 1);
        fut = sr.recv(rmsg, 1, 2);
    } else {
        sr.send(smsg, 0, 2);
        fut = sr.recv(rmsg, 0, 1);
    }

    int c = 0;
    do {
        c++;
    } while (fut.ready());

    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

    return rmsg;
}

auto test2_mesg() {
    mpi::communicator sr;

    mpi::message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::message<> rmsg{40, 40};

    bool arrived = false;

    if ( rank == 0 ) {
        auto fut = sr.send(smsg, 1, 1);
        sr.recv(rmsg, 1, 2, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    } else {
        auto fut = sr.send(smsg, 0, 2);
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    }

    int c = 0;
    do {
        c++;
        sr.progress();
     } while (!arrived);

    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

    return rmsg;
}

auto test1_shared_mesg() {
    mpi::communicator sr;

    mpi::shared_message<> smsg{10};
    for (int i = 0; i < 10; ++i) {
        smsg.enqueue(i);
    }

    mpi::shared_message<> rmsg{40, 40};

    mpi::communicator::common_future fut;

    if ( rank == 0 ) {
        auto sf = sr.send(smsg, 1, 1);
        fut = sr.recv(rmsg, 1, 2);
        sf.wait();
    } else {
        auto sf = sr.send(smsg, 0, 2);
        fut = sr.recv(rmsg, 0, 1);
        sf.wait();
    }

    int c = 0;
    do {
        c++;
     } while (fut.ready());

    std::cout << "\n***********\n";
    std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

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
        sr.recv(rmsg, 1, 2, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    } else {
        auto fut = sr.send(smsg, 0, 2);
        sr.recv(rmsg, 0, 1, [ &arrived](int src, int tag) {
            std::cout << src << ", " << tag << "\n";
            arrived = true;
        });
        fut.wait();
    }

    int c = 0;
    do {
        c++;
        sr.progress();
     } while (!arrived);

    std::cout << "\n***********\n";
        std::cout <<   "*" << std::setw(8) << c << " *\n";
    std::cout << "***********\n";

    return rmsg;
}


template <typename M>
bool check_msg(M const& msg) {
    std::cout << "Reference count " << msg.use_count() << " (size: " << msg.size() << ")\n";
    bool ok = true;

    for (int i = 0; i < msg.size()/sizeof(int); ++i) {
        std::cout << msg. template at<int>(i*sizeof(int)) << ", ";
        if ( msg. template at<int>(i*sizeof(int)) != i )
            ok = false;
    }
    std::cout << "\n";
    return ok;
}

bool check_msg(std::vector<unsigned char> msg) {
    bool ok = true;
    std::cout << " (size: " << msg.size() << ")\n";

    int c = 0;
    for (int i = 0; i < msg.size(); i += 4) {
        int value = *(reinterpret_cast<int*>(&msg[i]));
        std::cout << value << ", ";
        if ( value != c++ )
            ok = false;
    }
    std::cout << "\n";
    return ok;
}

template <typename Msg>
struct can_be_shared { static constexpr bool value = false; };

template <typename A>
struct can_be_shared<mpi::shared_message<A>> { static constexpr bool value = true; };

template <typename Test>
void run_test(Test&& test) {
    bool ok;
    auto msg = test();


    ok = check_msg(msg);

    std::cout << "Result: " << (ok?"PASSED":"FAILED") << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
}

int main(int argc, char** argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    run_test(test1);
    run_test(test2);


    run_test(test1_mesg);

    run_test(test2_mesg);

    run_test(test1_shared_mesg);

    run_test(test2_shared_mesg);

    MPI_Finalize();
}
