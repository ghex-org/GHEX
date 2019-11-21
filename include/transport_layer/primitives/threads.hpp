#ifndef _GHEX_TRANSPORT_LAYER_PRIMITIVES_THREADS_HPP_
#define _GHEX_TRANSPORT_LAYER_PRIMITIVES_THREADS_HPP_

#include <mutex>
//#include <experimental/barrier> not available yet
#include <condition_variable>
#include <atomic>
#include <vector>

struct threads {

    const int n_threads;
    std::mutex guard;
    std::mutex cv_guard;
    int barrier_cnt[2];
    std::atomic<int> counter;
    int up_counter[2];
    std::vector<std::condition_variable> cv, cv2;
    threads(int n)
        : n_threads{n}
        , barrier_cnt{n_threads, n_threads}
        , counter{n_threads}
        , up_counter{0,0}
        , cv(2)
        , cv2(2)
    {}

    class barrier_token {
        int epoch = 0;

        friend threads;
        barrier_token() {}
        void flip_epoch() { epoch = epoch ^ 1; }
        barrier_token(barrier_token const& ) = delete;
    public:
        barrier_token(barrier_token && ) = default;
    };

    barrier_token get_token() const { return {}; }

   void barrier(barrier_token& bt) {
        std::unique_lock<std::mutex> lock(cv_guard);

        barrier_cnt[bt.epoch]--;

        if (barrier_cnt[bt.epoch] == 0) {

            cv[bt.epoch].notify_all();
            cv2[bt.epoch].wait(lock, [this, &bt] { return barrier_cnt[bt.epoch] == n_threads;} );
        } else {
            cv[bt.epoch].wait(lock, [this, &bt] { return barrier_cnt[bt.epoch] == 0; });

            up_counter[bt.epoch]++;

            if (up_counter[bt.epoch] == n_threads-1) {
                up_counter[bt.epoch] = 0;
                barrier_cnt[bt.epoch] = n_threads; // done by multiple threads, but this resets the counter
                cv2[bt.epoch].notify_all();
            } else {
                cv2[bt.epoch].wait(lock, [this, &bt] { return barrier_cnt[bt.epoch] == n_threads;} );
            }
        }
        bt.flip_epoch();
    }

    template <typename F>
    void critical(F && f) {
        std::lock_guard<std::mutex> lock(guard);
        f();
    }

    template <typename F>
    void master(barrier_token& bt, F && f) { // Also this one should not be needed
        int x = counter.fetch_sub(1);
        if (x == 1) {
            f();
            counter = n_threads;
        }
        barrier(bt);
    }
};

#endif
