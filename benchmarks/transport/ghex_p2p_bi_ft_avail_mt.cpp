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
#ifdef USE_OPENMP
#include <omp.h>
#endif

#include <ghex/common/timer.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

#ifdef USE_UCP
// UCX backend
#include <ghex/transport_layer/ucx/context.hpp>
using transport    = ghex::tl::ucx_tag;
#else
// MPI backend
#include <ghex/transport_layer/mpi/context.hpp>
using transport    = ghex::tl::mpi_tag;
#endif

#include <ghex/transport_layer/message_buffer.hpp>
using context_type = ghex::tl::context<transport>;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::future<void>;

using MsgType = gridtools::ghex::tl::message_buffer<>;


#ifdef USE_OPENMP
std::atomic<int> sent(0);
std::atomic<int> received(0);
std::atomic<int> tail_send(0);
std::atomic<int> tail_recv(0);
#else
int sent(0);
int received(0);
int tail_send(0);
int tail_recv(0);
#endif

#ifdef USE_OPENMP
#define THREADID omp_get_thread_num()
#else
#define THREADID 0
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

#ifdef USE_OPENMP
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
        std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
        std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    {
        auto context_ptr = ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
        auto& context = *context_ptr;

#ifdef USE_OPENMP
#pragma omp parallel
#endif
        {
            auto comm              = context.get_communicator();
            const auto rank        = comm.rank();
            const auto size        = comm.size();
            const auto thread_id   = THREADID;
            const auto peer_rank   = (rank+1)%2;

            bool using_mt = false;
#ifdef USE_OPENMP
            using_mt = true;
#endif

            if (thread_id==0 && rank==0)
            {
                std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
            };

            std::vector<MsgType> smsgs(inflight);
            std::vector<MsgType> rmsgs(inflight);
            std::vector<future_type> sreqs(inflight);
            std::vector<future_type> rreqs(inflight);
            for(int j=0; j<inflight; j++)
            {
                smsgs[j].resize(buff_size);
                rmsgs[j].resize(buff_size);
                make_zero(smsgs[j]);
                make_zero(rmsgs[j]);
            }

#ifdef USE_OPENMP
#pragma omp single
#endif
            comm.barrier(MPI_COMM_WORLD);
#ifdef USE_OPENMP
#pragma omp barrier
#endif

            if(thread_id == 0)
            {
                timer.tic();
                ttimer.tic();
                if(rank == 1)
                    std::cout << "number of threads: " << num_threads << ", multi-threaded: " << using_mt << "\n";
            }

            int dbg = 0, sdbg = 0, rdbg = 0;
            int last_received = 0;
            int last_sent = 0;
            while(sent < niter || received < niter)
            {
                for(int j=0; j<inflight; j++)
                {
                    if(rank==0 && thread_id==0 && sdbg>=(niter/10)) {
                        std::cout << sent << " sent\n";
                        sdbg = 0;
                    }

                    if(rank==0 && thread_id==0 && rdbg>=(niter/10)) {
                        std::cout << received << " received\n";
                        rdbg = 0;
                    }

                    if(thread_id == 0 && dbg >= (niter/10)) {
                        dbg = 0;
                        std::cout << rank << " total bwdt MB/s:      "
                                  << ((double)(received-last_received + sent-last_sent)*size*buff_size/2)/timer.toc()
                                  << "\n";
                        timer.tic();
                        last_received = received;
                        last_sent = sent;
                    }

                    if(rreqs[j].test()) {
                        received++;
                        rdbg+=num_threads;
                        dbg+=num_threads;
                        rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight + j);
                    }

                    if(sent < niter && sreqs[j].test()) {
                        sent++;
                        sdbg+=num_threads;
                        dbg+=num_threads;
                        sreqs[j] = comm.send(smsgs[j], peer_rank, thread_id*inflight + j);
                    }
                }
            }

#ifdef USE_OPENMP
#pragma omp single
#endif
            comm.barrier(MPI_COMM_WORLD);
#ifdef USE_OPENMP
#pragma omp barrier
#endif

            if(thread_id == 0 && rank == 0){
                const auto t = ttimer.toc();
                std::cout << "time:       " << t/1000000 << "s\n";
                std::cout << "final MB/s: " << ((double)niter*size*buff_size)/t << "\n";
            }

            // tail loops - submit RECV requests until
            // all SEND requests have been finalized.
            // This is because UCX cannot cancel SEND requests.
            // https://github.com/openucx/ucx/issues/1162
            {
                int incomplete_sends = 0;
                int send_complete = 0;

                // complete all posted sends
                do {
                    comm.progress();
                    // check if we have completed all our posted sends
                    if(!send_complete){
                        incomplete_sends = 0;
                        for(int j=0; j<inflight; j++){
                            if(!sreqs[j].test()) incomplete_sends++;
                        }
                        if(incomplete_sends == 0) {
                            // increase thread counter of threads that are done with the sends
                            tail_send++;
                            send_complete = 1;
                        }
                    }
                    // continue to re-schedule all recvs to allow the peer to complete
                    for(int j=0; j<inflight; j++){
                        if(rreqs[j].test()) {
                            rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight + j);
                        }
                    }
                } while(tail_send!=num_threads);

                // We have all completed the sends, but the peer might not have yet.
                // Notify the peer and keep submitting recvs until we get his notification.
                future_type sf, rf;
                MsgType smsg(1), rmsg(1);

#ifdef USE_OPENMP
#pragma omp master
#endif
                {
                    sf = comm.send(smsg, peer_rank, 0x80000);
                    rf = comm.recv(rmsg, peer_rank, 0x80000);
                }

                while(tail_recv == 0){
                    comm.progress();

                    // schedule all recvs to allow the peer to complete
                    for(int j=0; j<inflight; j++){
                        if(rreqs[j].test()) {
                            rreqs[j] = comm.recv(rmsgs[j], peer_rank, thread_id*inflight + j);
                        }
                    }

#ifdef USE_OPENMP
#pragma omp master
#endif
                    {
                        if(rf.test()) tail_recv = 1;
                    }
                }
            }
            // peer has sent everything, so we can cancel all posted recv requests
            for(int j=0; j<inflight; j++){
                rreqs[j].cancel();
            }
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
