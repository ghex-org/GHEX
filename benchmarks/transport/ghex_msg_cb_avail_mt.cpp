#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>

#include "pmi.h"
#include "tictoc.h"
#include <omp.h>

#ifdef USE_MPI
#include "communicator_mpi.hpp"
using CommType = gridtools::ghex::mpi::communicator;
using namespace gridtools::ghex::mpi;
#define USE_CALLBACK_COMM
#else
#ifdef USE_UCX_NBR
#include "communicator_ucx_nbr.hpp"
#define USE_CALLBACK_COMM
#else
#include "communicator_ucx.hpp"
#undef  USE_CALLBACK_COMM
#endif
using CommType = gridtools::ghex::ucx::communicator;
using namespace gridtools::ghex::ucx;
#endif

/* TODO: this cannot be here, because it doesn't compile. I have to have it in the communicator hpp */
// extern CommType comm;
// DECLARE_THREAD_PRIVATE(comm)
// CommType comm;

#ifdef USE_CALLBACK_COMM

#include "callback_communicator.hpp"
extern gridtools::ghex::callback_communicator<CommType> comm_cb;
DECLARE_THREAD_PRIVATE(comm_cb)
gridtools::ghex::callback_communicator<CommType> comm_cb(comm);
#else
#define comm_cb comm
#endif

#include "message.hpp"
using MsgType = gridtools::ghex::mpi::raw_shared_message<>;

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

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   

    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;
    
    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << comm.name << "\n\n";

#pragma omp parallel
    {
	std::vector<MsgType> msgs;
	
	comm.init_mt();
	comm.whoami();

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size, buff_size);
	}
	
#pragma omp master
	if(rank == 1) {
	    tic();
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

	int i = 0, dbg = 0, blk;
	blk = niter / 10;
	dbg = dbg + blk;

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    while(total_sent < niter){
		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){

			// progress output
			if(rank==0 && thrid==0 && total_sent >= dbg) {
			    fprintf(stderr, "%d iters\n", total_sent);
			    dbg = dbg + blk;
			}

			// don't really need atomic, because we dont' really care about precise number of messages
			// as long as there are more than niter - we need to receive them
			total_sent++;

			// number of requests per thread
			submit_cnt++;

			available[thrid][j] = 0;
			comm_cb.send(msgs[j], 1, thrid*inflight+j, send_callback);
		    }
		}

		comm_cb.progress();
	    }

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
			comm_cb.recv(msgs[j], 0, thrid*inflight+j, recv_callback);
		    }
		}

		comm_cb.progress();
	    }	    
	}

#pragma omp barrier
#pragma omp master
	{
	    if(rank == 1) toc();
	    comm.fence();
	}

	printf("rank %d thread %d submitted %d serviced %d completion events, non-local %d\n", 
	       rank, thrid, submit_cnt, comm_cnt, nlcomm_cnt);
    }
}
