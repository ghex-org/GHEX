#include <atomic>
#include <mpi.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>

std::atomic<int> counter;

std::mutex io_serialize;


void safe_output(std::string const& s) {
    std::lock_guard<std::mutex> lock(io_serialize);
    std::cout << s << std::flush;
}

int threads_per_node = 3;

class gcl_object {
    int tag_generator; // unique value to generate tags

    int dst_proc;
    int dst_thread;
    int src_proc;
    int src_thread;

    int rank, m_size;
    int dtag;
    int stag;

    MPI_Request request;
public:
    gcl_object(int th_id, int dst_proc, int dst_thread, int src_proc, int src_thread)
        : tag_generator{counter.fetch_add(1)}
        , dst_proc{dst_proc}
        , dst_thread{dst_thread}
        , src_proc{src_proc}
        , src_thread{src_thread}
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_size);

        dtag = th_id + threads_per_node * rank;
        stag = src_thread + threads_per_node * src_proc;
        //        using namespace std::literals::chrono_literals;
        std::string log = "<" + std::to_string(rank) + ", " + std::to_string(th_id) + ">: "
            + std::to_string(tag_generator) + ": "
            + "s2r " + std::to_string(dst_proc)
            + ", t " + std::to_string(dst_thread) + " "
            + "tg " + std::to_string(dtag) + ", "
            + "rfr " + std::to_string(dst_proc)
            + ", t " + std::to_string(src_thread) + " "
            + "tg "+ std::to_string(stag) + "\n";
        safe_output(log);
    }

    void exchange(char const* payload, std::size_t size, char* buf) {

        MPI_Iecv(buf, size, MPI_BYTE, src_proc, stag, MPI_COMM_WORLD, &request);
        MPI_Request x;
        MPI_Isend(payload, size, MPI_BYTE, dst_proc, dtag, MPI_COMM_WORLD, &x);
    }

    void wait() {
        MPI_Status st;
        MPI_Wait(&request, &st);
    }
};


void exchange(int local_thread_id) {
    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    int global_thread_id = local_thread_id + threads_per_node * rank;

    gcl_object gcl(local_thread_id,
                   (local_thread_id==(threads_per_node-1))?(rank+1)%size:rank,
                   (local_thread_id==(threads_per_node-1))?0:local_thread_id+1,
                   (local_thread_id==0)?((rank==0)?(size-1):rank-1):rank,
                   (local_thread_id==0)?threads_per_node-1:local_thread_id-1);

    std::string payload("Hi, I am " + std::to_string(local_thread_id) +
                        " From processor " + std::to_string(rank));

    char resv[100];

    gcl.exchange(payload.c_str(), payload.length()+1, resv);
    gcl.wait();

    std::string res("<" + std::to_string(rank) + ", " + std::to_string(local_thread_id) + ">: " + resv + "\n");

    safe_output(res);
}

int main(int argc, char** argv) {
    MPI_Init_thread(&argc, &argv);

    std::thread t1(exchange, 0);
    std::thread t2(exchange, 1);
    std::thread t3(exchange, 2);

    t1.join();
    t2.join();
    t3.join();

    MPI_Finalize();
}
