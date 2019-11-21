#ifndef _GHEX_TRANSPORT_LAYER_PRIMITIVES_OPENMP_HPP_
#define _GHEX_TRANSPORT_LAYER_PRIMITIVES_OPENMP_HPP_

#ifdef _OPENMP
#include <omp.h>

struct threads {

    threads(int) {};

    class barrier_token {
        friend threads;
        barrier_token() {}
        barrier_token(barrier_token const& ) = delete;
    public:
        barrier_token(barrier_token && ) = default;
    };

    barrier_token get_token() const {return {};}

    static void barrier(barrier_token&) {
        #pragma omp barrier
    };
    template <typename F>
    static void critical(F && f) {
        #pragma omp critical
        f();
    }
    template <typename F>
    static void master(barrier_token&, F && f) { // Also this one should not be needed
        #pragma omp master
        f();
    }
};
#else
struct threads {


    threads(int) {};

    class barrier_token {
        friend threads;
        barrier_token() {}
        barrier_token(barrier_token const& ) = delete;
    public:
        barrier_token(barrier_token && ) = default;
    };


    static void barrier() {};
    template <typename F>
    static void critical(F && f) {
        f();
    }
    template <typename F>
    static void master(F && f) { // Also this one should not be needed
        f();
    }
};
#endif
#endif
