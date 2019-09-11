#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>
#include <vector>
#include <omp.h>
#include "tictoc.h"

#define USE_MPI
#ifdef USE_MPI
#include "communicator_mpi.hpp"
using CommType = gridtools::ghex::mpi::communicator;
#else
#include "communicator_ucx.hpp"
using CommType = gridtools::ghex::ucx::communicator;
#endif

CommType comm;


#include "message.hpp"
using MsgType = gridtools::ghex::mpi::shared_message<>;

/* available comm slots */
int *available = NULL;
int ongoing_comm = 0;

void send_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;
}

void recv_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;
}

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;
    printf("rank size %d %d\n", rank, size);

    {
	std::vector<MsgType> msgs;
	available = new int[inflight];

	for(int j=0; j<inflight; j++){
	    available[j] = 1;
	    msgs.emplace_back(buff_size, buff_size);
	}
	
	if(rank == 1) {
	    tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	if(rank == 0){

	    /* send niter messages - as slots become free */
	    int sent = 0;
	    while(sent != niter){

		for(int j=0; j<inflight; j++){
		    if(available[j]){
			if(rank==0 && (sent)%(niter/10)==0) fprintf(stderr, "%d iters\n", sent);
			available[j] = 0;
			sent++;
			ongoing_comm++;
			comm.send(msgs[j], 1, j, send_callback);
			if(sent==niter) break;
		    }
		}
		if(sent==niter) break;
	    
		/* progress a bit */
		int p = 0.1*inflight-1;
		do {
		    p-=comm.progress();
		} while(ongoing_comm && p>0);
	    }

	} else {

	    /* expect niter messages on receiver */
	    ongoing_comm = niter;

	    while(ongoing_comm){

		for(int j=0; j<inflight; j++){
		    if(available[j]){
			available[j] = 0;
			comm.recv(msgs[j], 0, j, recv_callback);
		    }
		}
	    
		/* progress a bit */
		int p = 0.1*inflight-1;
		do {
		    p-=comm.progress();
		} while(ongoing_comm && p>0);
	    }	    
	}

	/* complete all comm */
	while(ongoing_comm){
	    comm.progress();
	}

	if(rank == 1) toc();
    }
}
