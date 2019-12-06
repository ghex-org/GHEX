#include <iostream>
#include <vector>
#include <atomic>

#include <ghex/transport_layer/ucx/threads.hpp>
#include <ghex/common/timer.hpp>
#include "utils.hpp"

namespace ghex = gridtools::ghex;

#ifdef USE_MPI
    /* MPI backend */
    #ifdef USE_OPENMP
        #include <ghex/threads/atomic/primitives.hpp>
        using threading    = ghex::threads::atomic::primitives;
    #else
        #include <ghex/threads/none/primitives.hpp>
        using threading    = ghex::threads::none::primitives;
    #endif
    #include <ghex/transport_layer/mpi/context.hpp>
    using transport    = ghex::tl::mpi_tag;
#else
    /* UCX backend */
    #ifdef USE_OPENMP
        #include <ghex/threads/omp/primitives.hpp>
        using threading    = ghex::threads::omp::primitives;
    #else
        #include <ghex/threads/none/primitives.hpp>
        using threading    = ghex::threads::none::primitives;
    #endif
    #include <ghex/transport_layer/ucx3/address_db_mpi.hpp>
    #include <ghex/transport_layer/ucx/context.hpp>
    using db_type      = ghex::tl::ucx::address_db_mpi;
    using transport    = ghex::tl::ucx_tag;
#endif /* USE_MPI */
    
#include <ghex/transport_layer/message_buffer.hpp>
#include <ghex/transport_layer/shared_message_buffer.hpp>
using context_type = ghex::tl::context<transport, threading>;
//using MsgType = gridtools::ghex::tl::message_buffer<>;
using communicator_type = typename context_type::communicator_type;
using future_type = typename communicator_type::request_cb_type;

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


/* comm requests currently in-flight */
std::atomic<int> sent(0);
std::atomic<int> received(0);
std::atomic<int> tail_send(0);
std::atomic<int> tail_recv(0);
int last_received = 0;
int last_sent = 0;
int inflight;

/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt = 0, nlsend_cnt = 0, nlrecv_cnt = 0, submit_cnt = 0, submit_recv_cnt = 0;
DECLARE_THREAD_PRIVATE(comm_cnt)
DECLARE_THREAD_PRIVATE(nlsend_cnt)
DECLARE_THREAD_PRIVATE(nlrecv_cnt)
DECLARE_THREAD_PRIVATE(submit_cnt)
DECLARE_THREAD_PRIVATE(submit_recv_cnt)

int thrid, nthr;
DECLARE_THREAD_PRIVATE(thrid)
DECLARE_THREAD_PRIVATE(nthr)

void send_callback(communicator_type::message_type, int, int tag)
{
     std::cout << "send callback called " << /*rank <<*/ /*" thread " << omp_get_thread_num() <<*/ " tag " << tag << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlsend_cnt++;
    comm_cnt++;
    sent++;
}

void recv_callback(communicator_type::message_type, int, int tag)
{
     std::cout << "recv callback called " << /*rank <<*/ /*" thread " << omp_get_thread_num() <<*/ " tag " << tag /*<< " ongoing " << ongoing_comm*/ << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlrecv_cnt++;
    comm_cnt++;
    received++;
}

int main(int argc, char *argv[])
{
    int niter, buff_size;
    int mode;

    if(argc != 4){
	std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
	std::terminate();
    }
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   

    gridtools::ghex::timer timer, ttimer;
    
    int num_threads = 1;
#ifdef USE_OPENMP
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
    #pragma omp parallel
    {
        #pragma omp master
        num_threads = omp_get_num_threads();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
    
    {
        #ifdef USE_MPI
        context_type contex{num_threads, MPI_COMM_WORLD};
        #else
        context_type context{num_threads, MPI_COMM_WORLD, db_type{MPI_COMM_WORLD} };
        #endif


        THREAD_PARALLEL_BEG() {

            auto token = context.get_token();

            auto comm = context.get_communicator(token);

            const auto rank = comm.rank();
            const auto size = comm.size();
            const auto thread_id = token.id();
            const auto num_threads = context.thread_primitives().size();
            const auto peer_rank = (rank+1)%2;
            thrid = token.id();
            nthr = num_threads;
            
            context.thread_primitives().master(token, 
                [rank,&comm]()
                {
                    if(rank==0)
                    std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
                });
            
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
            
            context.barrier(token);
            
            context.thread_primitives().master(token, 
                [&timer,&ttimer,rank,num_threads]() 
                { 
                    timer.tic();
                    ttimer.tic();
                    if(rank == 1) 
                        std::cout << "number of threads: " << num_threads << ", multi-threaded: " << THREAD_IS_MT << "\n";
                });

            context.thread_primitives().barrier(token);
            
            // send/recv niter messages - as soon as a slot becomes free
            int dbg = 0, sdbg = 0, rdbg = 0;
            while(sent < niter || received < niter)
            {
                if(thrid == 0 && dbg >= (niter/10)) {
                    dbg = 0;
                    std::cout << rank << " total bwdt MB/s:      " 
                              << ((double)(received-last_received + sent-last_sent)*size*buff_size/2)/timer.toc()
                              << "\n";
                    timer.tic();
                    last_received = received;
                    last_sent = sent;
                }

                if(rank==0 && thrid==0 && rdbg >= (niter/10)) {
                    std::cout << received << " received\n";
                    rdbg = 0;
                }

                if(rank==0 && thrid==0 && sdbg >= (niter/10)) {
                    std::cout << sent << " sent\n";
                    sdbg = 0;
                }
                for(int j=0; j<inflight; j++){
                    if(rmsgs[j].use_count() == 1){
                        submit_recv_cnt += nthr;
                        rdbg += nthr;
                        dbg += nthr;
                        auto msg = rmsgs[j];
                        rreqs[j] = comm.recv(std::move(msg), peer_rank, thrid*inflight+j, recv_callback);
                    } else comm.progress();

                    if(sent < niter && smsgs[j].use_count() == 1){
                        submit_cnt += nthr;
                        sdbg += nthr;
                        dbg += nthr;
                        auto msg = smsgs[j];
                        sreqs[j] = comm.send(std::move(msg), peer_rank, thrid*inflight+j, send_callback);
                    } else comm.progress();
                }
            }
            
            context.barrier(token);
            context.thread_primitives().master(token, 
                [&niter,size,buff_size,&ttimer,rank,num_threads]() 
                { 
                    if (rank==1)
                    {
                        const auto t = ttimer.toc();
                        std::cout << "time:       " << t/1000000 << "s\n";
                        std::cout << "final MB/s: " << ((double)niter*size*buff_size)/t << "\n";
                    }       
                });
            
            context.thread_primitives().barrier(token);

            context.thread_primitives().critical(
                [&]()
                {
                    std::cout << "rank " << rank << " thread " << thrid << " sends submitted " << submit_cnt/nthr
                              << " serviced " << comm_cnt << ", non-local sends " << nlsend_cnt << " non-local recvs " << nlrecv_cnt << "\n";

                });

            // tail loops - submit RECV requests until
            // all SEND requests have been finalized.
            // This is because UCX cannot cancel SEND requests.
            // https://github.com/openucx/ucx/issues/1162
            // 
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
                            auto msg = rmsgs[j];
                            rreqs[j] = comm.recv(std::move(msg), peer_rank, thread_id*inflight + j, recv_callback);
                        }
                    }
                } while(tail_send!=num_threads);
                
                // We have all completed the sends, but the peer might not have yet.
                // Notify the peer and keep submitting recvs until we get his notification.
                future_type sf, rf;
                MsgType smsg(1), rmsg(1);
                context.thread_primitives().master(token, 
                    [&comm,&sf,&rf,smsg,rmsg,peer_rank]() mutable
                    {
                        sf = comm.send(std::move(smsg), peer_rank, 0x800000, [](communicator_type::message_type, int, int){});
                        rf = comm.recv(std::move(rmsg), peer_rank, 0x800000, [](communicator_type::message_type, int, int){});
                    });

                while(!tail_recv.load()){
                    comm.progress();

                    // schedule all recvs to allow the peer to complete
                    for(int j=0; j<inflight; j++){
                        if(rreqs[j].test()) {
                            auto msg = rmsgs[j];
                            rreqs[j] = comm.recv(std::move(msg), peer_rank, thread_id*inflight + j, recv_callback);
                        }
                    }
                    context.thread_primitives().master(token, 
                        [&rf,&tail_recv]()
                        {
                            if(rf.test()) tail_recv = 1;
                        });
                }
            }
            // peer has sent everything, so we can cancel all posted recv requests
            for(int j=0; j<inflight; j++){
                rreqs[j].cancel();
            }

        } THREAD_PARALLEL_END();    
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
