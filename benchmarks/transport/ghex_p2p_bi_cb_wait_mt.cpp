#include <iostream>
#include <vector>
#include <atomic>

#include <ghex/transport_layer/ucx/threads.hpp>
#include <ghex/common/timer.hpp>
#include "utils.hpp"


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
#else

/* UCX backend */
#ifdef USE_UCX_NBR
#include <ghex/transport_layer/callback_communicator.hpp>
#else
#include <ghex/transport_layer/ucx/callback_communicator.hpp>
#endif
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
#endif /* USE_MPI */

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


/* comm requests currently in-flight */
std::atomic<int> sent = 0;
std::atomic<int> received = 0;

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


void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlsend_cnt++;
    comm_cnt++;
    sent++;
}

void recv_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlrecv_cnt++;
    comm_cnt++;
    received++;
}

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;

    gridtools::ghex::timer timer;
    long bytes = 0;

#ifdef USE_MPI
    int mode;
#ifdef THREAD_MODE_MULTIPLE
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
#endif

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   

    THREAD_PARALLEL_BEG() {
	gridtools::ghex::tl::callback_communicator<CommType> comm;

	THREAD_MASTER() {
	    rank = comm.rank();
	    size = comm.size();
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
	}

	thrid = GET_THREAD_NUM();
	nthr = GET_NUM_THREADS();

	std::vector<MsgType> smsgs;
	std::vector<MsgType> rmsgs;
	
	for(int j=0; j<inflight; j++){
	    smsgs.emplace_back(buff_size);
	    rmsgs.emplace_back(buff_size);
    	    make_zero(smsgs[j]);
    	    make_zero(rmsgs[j]);
	}
	
	comm.barrier();

	THREAD_MASTER() {
	    if(rank == 1) {
		std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
		timer.tic();
		bytes = (double)niter*size*buff_size;
	    }
	}

	
	/* send / recv niter messages, work in inflight requests at a time */
	int i = 0, dbg = 0;
	while(i<niter){

	    THREAD_BARRIER();

	    /* submit inflight requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && dbg >= (niter/10)) {
		    std::cout << i << " iters\n";
		    dbg = 0;
		}
		dbg+=nthr;
		i+=nthr;
		comm.send(smsgs[j], peer_rank, thrid*inflight+j, send_callback);
		comm.recv(rmsgs[j], peer_rank, thrid*inflight+j, recv_callback);
	    }
	
	    /* complete all inflight requests before moving on */
	    while(sent < nthr*inflight || received < nthr*inflight){
	    	comm.progress();
	    }

	    THREAD_BARRIER();

	    sent = 0;
	    received = 0;
	}

	comm.barrier();
	comm.finalize();

#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " sends submitted " << submit_cnt/nthr
		  << " serviced " << comm_cnt << ", non-local sends " << nlsend_cnt << " non-local recvs " << nlrecv_cnt << "\n";

    } THREAD_PARALLEL_END();
    
    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
#endif
}
