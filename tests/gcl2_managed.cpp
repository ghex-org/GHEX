#include <atomic>
#include <mpi.h>
#include <iostream>
#include <thread>
#include <mutex>
#include <string>
#include <chrono>
#include <future>
#include <memory>
#include <queue>
#include <list>
#include <prototype/facilities.hpp>
#include <sstream>

void gcl2_init(int &argc, char**& argv) {
    MPI_Init(&argc, &argv);
}

std::atomic<bool> terminate; // it is not necessary to be atomic, but it must be guaranteed to be seen by all threads at some point

struct job {
    char const *send_buf;
    char *recv_buf;
    std::size_t size;
    int src_proc, dst_proc;
    int stag, dtag;

    job(char const* sb, char* rb, std::size_t s, int sp, int dp, int st, int rt)
        : send_buf{sb}
        , recv_buf{rb}
        , size{s}
        , src_proc{sp}
        , dst_proc{dp}
        , stag{st}
        , dtag{rt}
    {}
};

struct job_q {
    char const *send_buf;
    char *recv_buf;
    std::size_t size;
    int src_proc, dst_proc;
    int stag, dtag;
    std::promise<void> prom;
    MPI_Request request;

    job_q(job j)
        : send_buf{j.send_buf}
        , recv_buf{j.recv_buf}
        , size{j.size}
        , src_proc{j.src_proc}
        , dst_proc{j.dst_proc}
        , stag{j.stag}
        , dtag{j.dtag}
        , prom{}
        , request{}
    {}

    job_q(job_q &&j)
        : send_buf{j.send_buf}
        , recv_buf{j.recv_buf}
        , size{j.size}
        , src_proc{j.src_proc}
        , dst_proc{j.dst_proc}
        , stag{j.stag}
        , dtag{j.dtag}
        , prom{std::move(j.prom)}
        , request{j.request}
    {}
};

std::mutex queue_m;
std::queue<job_q> send_queue;
std::list<job_q> wait_queue;

/* Main function to make the communication happen. It enqueue the message to be sent anf the pointer to where put the received one.
   Returns a future to check for completion. */
std::future<void> push(job j)
{
    {
        std::lock_guard<std::mutex> lock(queue_m);
        send_queue.push(j);
        return send_queue.back().prom.get_future();
    }
}

// Main thread polling the queue of incoming messages to send/recv
void sender () {
    while (!terminate) {
        while (!send_queue.empty()) {
            auto& j = send_queue.front();
            MPI_Request request;
            MPI_Irecv(j.recv_buf, j.size, MPI_BYTE, j.src_proc, j.stag, MPI_COMM_WORLD, &j.request);
            MPI_Request x;
            MPI_Isend(j.send_buf, j.size, MPI_BYTE, j.dst_proc, j.dtag, MPI_COMM_WORLD, &x);
            queue_m.lock();
            wait_queue.push_back(std::move(j));
            send_queue.pop();
            queue_m.unlock();
        }
    }
}

// Thread waiting for the completion of the operation, it's the one fullfilling the promise
void waiter () {
    while (!terminate) {
        while (!wait_queue.empty()) {
            auto begin = wait_queue.begin();
            auto it = begin;
            queue_m.lock();
            auto end = --wait_queue.end();
            queue_m.unlock();
            for ( ; it != end; ++it) {
                MPI_Status st;
                int res;
                MPI_Test(&(it->request), &res, &st);
                if (res) { // Received
                    it->prom.set_value();
                    queue_m.lock();
                    wait_queue.erase(it);
                    queue_m.unlock();
                }
            }
            MPI_Status st;
            int res;
            MPI_Test(&(it->request), &res, &st);
            if (res) { // Received
                it->prom.set_value();
                queue_m.lock();
                wait_queue.erase(it);
                queue_m.unlock();
            }
        }
    }
}


// Runtime system
struct gcl2_rt {
    std::thread send_t, wait_t;

    gcl2_rt() : send_t{sender}, wait_t{waiter}
    {
        terminate = false;
        safe_output("Runtime Started\n");
    }

    ~gcl2_rt() {
        terminate = true;
        send_t.join();
        wait_t.join();
        safe_output("Runtime Terminated\n");
}
};

gcl2_rt rt;

class gcl_object {
    int tag_generator; // unique value to generate tags

    int dst_proc;
    int dst_thread;
    int src_proc;
    int src_thread;

    int rank, m_size;
    int dtag;
    int stag;

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

        dtag = (th_id + threads_per_node * rank) * 100 + tag_generator;
        stag = (src_thread + threads_per_node * src_proc) * 100 + tag_generator;

        std::stringstream tmp;
        tmp << std::this_thread::get_id();

        std::string log = "<" + std::to_string(rank) + ", " + std::to_string(th_id) + ">: "
            + tmp.str() + " "
            + "obj_counter " + std::to_string(tag_generator) + " "
            + "s2r " + std::to_string(dst_proc)
            + ", t " + std::to_string(dst_thread) + " "
            + "tg " + std::to_string(dtag) + ", "
            + "rfr " + std::to_string(dst_proc)
            + ", t " + std::to_string(src_thread) + " "
            + "tg "+ std::to_string(stag) + "\n";
        safe_output(log);
    }

    std::future<void> exchange(char const* payload, std::size_t size, char* buf) {

        return push(job{payload, buf, size, src_proc, dst_proc, stag, dtag});
    }

};
