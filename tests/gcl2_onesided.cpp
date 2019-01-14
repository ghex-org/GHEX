#include <atomic>
#include <mpi.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include <future>
#include <prototype/facilities.hpp>
#include <sstream>

void gcl2_init(int &argc, char**& argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);
}


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

    struct future {
        MPI_Request request;

        future(MPI_Request x) : request{x} {}

        void get() {
            MPI_Status st;
            MPI_Wait(&request, &st);
        }
    };
public:
    gcl_object(int th_id, int dst_proc, int dst_thread, int src_proc, int src_thread)
        : tag_generator{object_id()}
        , dst_proc{dst_proc}
        , dst_thread{dst_thread}
        , src_proc{src_proc}
        , src_thread{src_thread}
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &m_size);

        dtag = (th_id + threads_per_node * rank) * 100 + tag_generator;;
        stag = (src_thread + threads_per_node * src_proc) * 100 + tag_generator;;

        std::stringstream tmp;
        tmp << std::this_thread::get_id();

        std::string log = "<" + std::to_string(rank) + ", " + std::to_string(th_id) + ">: "
            + tmp.str() + " "
            + ": obj_counter " + std::to_string(tag_generator) + " "
            + "s2r " + std::to_string(dst_proc)
            + ", t " + std::to_string(dst_thread) + " "
            + "tg " + std::to_string(dtag) + ", "
            + "rfr " + std::to_string(dst_proc)
            + ", t " + std::to_string(src_thread) + " "
            + "tg "+ std::to_string(stag) + "\n";
        safe_output(log);
    }

    future exchange(char const* payload, std::size_t size, char* buf) {

        MPI_Irecv(buf, size, MPI_BYTE, src_proc, stag, MPI_COMM_WORLD, &request);
        MPI_Request x;
        MPI_Isend(payload, size, MPI_BYTE, dst_proc, dtag, MPI_COMM_WORLD, &x);
        return request;
    }

};
