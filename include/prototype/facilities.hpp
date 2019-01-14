#include <map>
#include <atomic>
#include <mutex>
#include <iostream>
#include <condition_variable>

std::mutex critical;

std::map<std::thread::id, int> track;

int object_id() {
    std::lock_guard<std::mutex> lock(critical);
    if (track.find(std::this_thread::get_id()) == track.end()) {
        // first one
        track[std::this_thread::get_id()] = 1;;
    } else {
        ++track[std::this_thread::get_id()];
    }

    return track[std::this_thread::get_id()];
}

std::mutex io_serialize;

void safe_output(std::string const& s) {
    std::lock_guard<std::mutex> lock(io_serialize);
    std::cout << s << std::flush;
}

const int threads_per_node = 3;

int m_barrier = threads_per_node-1;
std::condition_variable m_cv;
std::mutex m_cv_m;
int m_generation = 0;

void gcl2_barrier() {
    std::unique_lock<std::mutex> lk{m_cv_m};
    int gen = m_generation;
    if (!--m_barrier) {
        MPI_Barrier(MPI_COMM_WORLD);
        m_barrier = threads_per_node-1;
        m_generation++;
        m_cv.notify_all();
    } else {
        m_cv.wait(lk, [&]() {
                return m_generation!=gen;
            });
    }
}
