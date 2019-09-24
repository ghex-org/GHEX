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

int ongoing_comm = 0;
int inflight;

void send_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;

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

	comm_cnt = 0;
	nlcomm_cnt = 0;
	submit_cnt = 0;
	int i = 0, dbg = 0;

	/* send / recv niter messages, work in inflight requests at a time */
	while(i<niter){

#pragma omp atomic
	    ongoing_comm += inflight;

	    /* submit inflight requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && dbg>=(niter/10)) {fprintf(stderr, "%d iters\n", i); dbg=0;}
		submit_cnt++;
		i += nthr;
		dbg += nthr; 
		if(rank==0)
		    comm.send(msgs[j], 1, thrid*inflight+j, send_callback);
		else
		    comm.recv(msgs[j], 0, thrid*inflight+j, recv_callback);
		if(i >= niter) break;
	    }

	    /* complete all inflight requests before moving on */
	    while(ongoing_comm){
		comm.progress();
	    }

	    /* have to have the barrier since any thread can complete any request in UCX */
#pragma omp barrier
	}

#pragma omp master
	{
	    if(rank == 1) toc();
	    comm.fence();
	}

	printf("rank %d thread %d submitted %d serviced %d completion events, non-local %d\n", 
	       rank, thrid, submit_cnt, comm_cnt, nlcomm_cnt);
    }
}
