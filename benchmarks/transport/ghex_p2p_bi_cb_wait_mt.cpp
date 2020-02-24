/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#include <iostream>
#include <vector>
#include <atomic>

#include <ghex/common/timer.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

#ifdef USE_OPENMP
#include <ghex/threads/omp/primitives.hpp>
using threading    = ghex::threads::omp::primitives;
#else
#include <ghex/threads/none/primitives.hpp>
using threading    = ghex::threads::none::primitives;
#endif

#ifdef USE_UCP
// UCX backend
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;
#else
// MPI backend
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

#include <ghex/transport_layer/shared_message_buffer.hpp>
using context_type = ghex::tl::context<transport, threading>;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::request_cb_type;

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


#ifdef USE_OPENMP
std::atomic<int> sent(0);
std::atomic<int> received(0);
#else
int sent;
int received;
#endif

int main(int argc, char *argv[])
{
    int niter, buff_size;
    int inflight;
    int mode;
    gridtools::ghex::timer timer, ttimer;

    if(argc != 4)
    {
        std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
        std::terminate();
    }
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    int num_threads = 1;

#ifdef USE_OPENMP
#pragma omp parallel
    {
#pragma omp master
        num_threads = omp_get_num_threads();
    }
#endif

    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);

    {
        auto context_ptr = ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD);
        auto& context = *context_ptr;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            auto token             = context.get_token();
            auto comm              = context.get_communicator(token);
            const auto rank        = comm.rank();
            const auto size        = comm.size();
            const auto thread_id   = token.id();
            const auto num_threads = context.thread_primitives().size();
            const auto peer_rank   = (rank+1)%2;

            bool using_mt = false;
#ifdef USE_OPENMP
            using_mt = true;
#endif

            int comm_cnt = 0, nlsend_cnt = 0, nlrecv_cnt = 0;

            auto send_callback = [&](communicator_type::message_type, int, int tag)
            {
                // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
                int pthr = tag/inflight;
                if(pthr != thread_id) nlsend_cnt++;
                comm_cnt++;
                sent++;
            };

            auto recv_callback = [&](communicator_type::message_type, int, int tag)
            {
                // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
                int pthr = tag/inflight;
                if(pthr != thread_id) nlrecv_cnt++;
                comm_cnt++;
                received++;
            };

            if (thread_id==0 && rank==0)
            {
                if(rank==0)     std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
            }

            std::vector<MsgType> smsgs;
            std::vector<MsgType> rmsgs;
            std::vector<future_type> sreqs;
            std::vector<future_type> rreqs;

            for(int j=0; j<inflight; j++){
                smsgs.emplace_back(buff_size);
                rmsgs.emplace_back(buff_size);
                make_zero(smsgs[j]);
                make_zero(rmsgs[j]);
            }
            sreqs.resize(inflight);
            rreqs.resize(inflight);

            comm.barrier();

            if (thread_id == 0)
            {
                timer.tic();
                ttimer.tic();
                if(rank == 1)
                    std::cout << "number of threads: " << num_threads << ", multi-threaded: " << using_mt << "\n";
            }

            // send / recv niter messages, work in inflight requests at a time
            int i = 0, dbg = 0;
            int last_i = 0;
            while(i<niter){

#ifdef USE_OPENMP
#pragma omp barrier
#endif
                if(thread_id == 0 && dbg >= (niter/10)) {
                    dbg = 0;
                    std::cout << rank << " total bwdt MB/s:      "
                              << ((double)(i-last_i)*size*buff_size)/timer.stoc()
                              << "\n";
                    timer.tic();
                    last_i = i;
                }

                // submit inflight requests
                for(int j=0; j<inflight; j++){
                    dbg+=num_threads;
                    i+=num_threads;
                    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight+j, recv_callback);
                    sreqs[j] = comm.send(smsgs[j], peer_rank, thread_id*inflight+j, send_callback);
                }

                // complete all inflight requests before moving on
                while(sent < num_threads*inflight || received < num_threads*inflight){
                    comm.progress();
                }

#ifdef USE_OPENMP
#pragma omp barrier
#endif
                sent = 0;
                received = 0;
            }

            comm.barrier();
            if(thread_id==0 && rank == 0)
            {
                const auto t = ttimer.stoc();
                std::cout << "time:       " << t/1000000 << "s\n";
                std::cout << "final MB/s: " << ((double)niter*size*buff_size)/t << "\n";
            }

            // stop here to help produce a nice std output
            comm.barrier();
            context.thread_primitives().critical(
                [&]()
                {
                    std::cout
                    << "rank " << rank << " thread " << thread_id << " serviced " << comm_cnt
                    << ", non-local sends " << nlsend_cnt << " non-local recvs " << nlrecv_cnt << "\n";
                });

            // tail loops - not needed in wait benchmarks
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
