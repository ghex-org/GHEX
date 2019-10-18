#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>
#include <omp.h>

/* define to use the raw shared message - lower overhead */
#define GHEX_USE_RAW_SHARED_MESSAGE

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/callback_communicator.hpp>
using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
#define USE_CALLBACK_COMM
#else

/* UCX backend */
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;

#ifdef USE_UCX_NBR
/* use the GHEX callback framework */
#define USE_CALLBACK_COMM
#else
/* use the UCX's own callback framework */
#include <ghex/transport_layer/ucx/communicator.hpp>
#undef  USE_CALLBACK_COMM
#endif /* USE_UCX_NBR */

#endif /* USE_MPI */


/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt, nlcomm_cnt;
int submit_cnt;
int thrid, nthr;
#pragma omp threadprivate(comm_cnt, nlcomm_cnt, submit_cnt, thrid, nthr)

/* available comm slots, one array per thread:
   MUST BE SHARED and long enough so that all threads can keep their
   message status there. Note: other threads can complete the request.
 */
int **available = NULL;

int total_sent = 0;
int ongoing_comm = 0;
int inflight;

void send_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    available[pthr][pos] = 1;
    comm_cnt++;
}

void recv_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << " pthr " <<  pthr << " ongoing " << ongoing_comm << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    available[pthr][pos] = 1;

#pragma omp atomic
    ongoing_comm--;
    comm_cnt++;
}

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;

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

#ifndef USE_MPI
    std::cout << "ghex request size: " << CommType::get_request_size<MsgType>() << "\n";
#endif	

#pragma omp parallel
    {
	/* TODO this needs to be made per-thread. 
	   If we make 'static' variables, then we can't initialize m_rank and anything else
	   that used MPI in the constructor, as it will be executed before MPI_Init.
	*/
	CommType *comm = new CommType();

#ifdef USE_CALLBACK_COMM
	gridtools::ghex::tl::callback_communicator<CommType> comm_cb(*comm);
#else
#define comm_cb (*comm)
#endif

#pragma omp master
	{
	    rank = comm->m_rank;
	    size = comm->m_size;
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(*comm).name() << "\n\n";
	}

	gridtools::ghex::timer timer;
	long bytes = 0;
	std::vector<MsgType> msgs;
	
	comm->init_mt();
#pragma omp barrier
	comm->whoami();

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size);
	}
	
#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	/** initialize request availability arrays */
#pragma omp master
	available = new int *[nthr];

#pragma omp barrier
#pragma omp flush(available)
	available[thrid] = new int[inflight];
	for(int j=0; j<inflight; j++){
	    available[thrid][j] = 1;
	}

	/* just in case, make sure all threads have their arrays initialized */
#pragma omp barrier

	comm_cnt = 0;
	nlcomm_cnt = 0;
	submit_cnt = 0;

#define NOPROGRESS_CNT 10000
	int i = 0, dbg = 0, blk, noprogress = 0;
	blk = niter / 10;
	dbg = dbg + blk;

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    while(total_sent < niter && noprogress < NOPROGRESS_CNT){
		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){

			// reset the no-progress counter
			noprogress = 0;

			// progress output
			if(rank==0 && thrid==0 && total_sent >= dbg) {
			    std::cout << total_sent << " iters\n";
			    dbg = dbg + blk;
			}

			// don't really need atomic, because we dont' really care about precise number of messages
			// as long as there are more than niter - we need to receive them
			total_sent++;

			// number of requests per thread
			submit_cnt++;

			available[thrid][j] = 0;
			comm_cb.send(1, thrid*inflight+j, msgs[j], send_callback);
		    }
		}
		comm_cb.progress();
		noprogress++;
	    }
	    if(noprogress >= NOPROGRESS_CNT) std::cout << "sender finished: no progress threashold\n";

	} else {

	    /* recv requests are resubmitted as soon as a request is completed */
	    /* so the number of submitted recv requests is always ~constant (inflight) */
	    /* expect niter messages (i.e., niter recv callbacks) on receiver  */
	    ongoing_comm = niter;
#pragma omp barrier

	    while(ongoing_comm>0){

		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){

			// number of requests per thread
			submit_cnt++;

			available[thrid][j] = 0;
			comm_cb.recv(0, thrid*inflight+j, msgs[j], recv_callback);
		    }
		}

		comm_cb.progress();
	    }	    
	}

#pragma omp barrier
#pragma omp master
	{
	    if(rank == 1) timer.vtoc(bytes);
	    // comm.fence();
	}

#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " submitted " << submit_cnt
		  << " serviced " << comm_cnt << ", non-local " << nlcomm_cnt << " completion events\n";
    }
}
