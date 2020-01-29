#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/threads/omp/primitives.hpp>
#include <iostream>
#include <iomanip>

#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using threading = gridtools::ghex::threads::std_thread::primitives;
using threading2 = gridtools::ghex::threads::atomic::primitives;
using threading3 = gridtools::ghex::threads::omp::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

TEST(transport, barrier_mt_std) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(4, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    auto func = [&context]() {
        auto token = context.get_token();
        auto comm = context.get_communicator(token);

        for(int i=0; i<50; i++)  {
            comm.barrier();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(4);
    for (int i=0; i<4; ++i)
        threads.push_back(std::thread{func});
    for (auto& t : threads)
        t.join();
}

TEST(transport, barrier_mt_std_atomic) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading2>::create(4, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    
    auto func = [&context]() {
        auto token = context.get_token();
        auto comm = context.get_communicator(token);

        for(int i=0; i<50; i++)  {
            comm.barrier();
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(4);
    for (int i=0; i<4; ++i)
        threads.push_back(std::thread{func});
    for (auto& t : threads)
        t.join();
}

TEST(transport, barrier_mt_omp) {
    int num_threads = 1;
    omp_set_num_threads(4);
#pragma omp parallel
#pragma omp master
    num_threads = omp_get_num_threads();

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading3>::create(num_threads, MPI_COMM_WORLD);
    auto& context = *context_ptr;
#pragma omp parallel
    {
        auto token = context.get_token();
        auto comm = context.get_communicator(token);

        for(int i=0; i<50; i++)  {
            comm.barrier();
        }
    }
}

TEST(transport, barrier_mt_omp_atomic) {
    int num_threads = 1;
    omp_set_num_threads(4);
#pragma omp parallel 
#pragma omp master
    num_threads = omp_get_num_threads();

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading2>::create(num_threads, MPI_COMM_WORLD);
    auto& context = *context_ptr;
#pragma omp parallel
    {
        auto token = context.get_token();
        auto comm = context.get_communicator(token);

        for(int i=0; i<50; i++)  {
            comm.barrier();
        }
    }
}
