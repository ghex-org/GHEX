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
int ongoing_comm = 0;

void send_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    ongoing_comm--;
}

void recv_callback(int rank, int tag, MsgType &mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
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

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size, buff_size);
	}
	
	if(rank == 1) {
	    tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	/* send / recv niter messages, work in inflight requests at a time */
	int i = 0;
	while(i<niter){

	    /* submit inflight requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && (i)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);
		i++;
		ongoing_comm++;
		if(rank==0)
		    comm.send(msgs[j], 1, j, send_callback);
		else
		    comm.recv(msgs[j], 0, j, recv_callback);
		if(i==niter) break;
	    }
		
	    /* complete all inflight requests before moving on */
	    while(ongoing_comm){
		comm.progress();
	    }
	}

	if(rank == 1) toc();
    }
}
