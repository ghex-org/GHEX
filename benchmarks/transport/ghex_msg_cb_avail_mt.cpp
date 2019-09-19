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
#else
#ifdef USE_UCX_NBR
#include "communicator_ucx_nbr.hpp"
#else
#include "communicator_ucx.hpp"
#endif
using CommType = gridtools::ghex::ucx::communicator;
using namespace gridtools::ghex::ucx;
#endif

#include "message.hpp"
using MsgType = gridtools::ghex::mpi::shared_message<>;

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

int ongoing_comm = 0;
int inflight;

void send_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    available[pthr][pos] = 1;

#pragma omp atomic
    ongoing_comm--;
    comm_cnt++;
}

void recv_callback(int rank, int tag, MsgType &mesg)
{
    //std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
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
	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);

#pragma omp master
	available = new int *[nthr];

#pragma omp barrier
#pragma omp flush(available)
	available[thrid] = new int[inflight];
	for(int j=0; j<inflight; j++){
	    available[thrid][j] = 1;
	}
#pragma omp barrier

	comm_cnt = 0;
	nlcomm_cnt = 0;
	submit_cnt = 0;
	int i = 0, dbg = 0;

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    int sent = 0;
	    while(sent < niter){

		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){
			if(rank==0 && thrid==0 && dbg>=(niter/10)) {fprintf(stderr, "%d iters\n", sent); dbg=0;}
#pragma omp atomic
			ongoing_comm++;
			available[thrid][j] = 0;
			comm.send(msgs[j], 1, thrid*inflight+j, send_callback);
			submit_cnt++;
			dbg  += nthr; 
			sent += nthr; 
			if(sent>=niter) break;
		    }
		}
		if(sent>=niter) break;
	    
		/* progress a bit: for large inflight values this yields better performance */
		/* over simply calling the progress once */
		int p = 0.1*inflight-1;
		do {
		    p-=comm.progress();
		} while(ongoing_comm>0 && p>0);
	    }

	} else {

	    /* recv requests are resubmitted as soon as a request is completed */
	    /* so the number of submitted recv requests is always constant (inflight) */
	    /* expect niter messages (i.e., niter recv callbacks) on receiver  */
#pragma omp master
	    ongoing_comm = niter;

#pragma omp barrier
#pragma omp flush(ongoing_comm)

	    while(ongoing_comm){

		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){
			available[thrid][j] = 0;
			comm.recv(msgs[j], 0, thrid*inflight+j, recv_callback);
			submit_cnt++;
		    }
		}

		/* progress a bit: for large inflight values this yields better performance */
		/* over simply calling the progress once */
		int p = 0.1*inflight-1;
		do {
		    p-=comm.progress();
		} while(ongoing_comm>0 && p>0);
	    }	    
	}

	/* complete all comm */
	while(ongoing_comm){
	    comm.progress();
	}

#pragma omp barrier
#pragma omp master
	if(rank == 1) toc();
	printf("rank %d thread %d submitted %d serviced %d completion events, non-local %d\n", rank, thrid, submit_cnt, comm_cnt, nlcomm_cnt);

	comm.fence();
    }
}
