#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

#include "tictoc.h"
#include "pmi.h"
#include "communicator_ucx.hpp"
using CommType = gridtools::ghex::ucx::communicator;
using namespace gridtools::ghex::ucx;

int ongoing_comm = 0;

void send_callback(int rank, int tag, void *mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    ongoing_comm--;
}

void recv_callback(int rank, int tag, void *mesg)
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

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << comm.name << "\n\n";    

    {
	unsigned char **buffers = new unsigned char *[inflight];

	for(int j=0; j<inflight; j++){
	    posix_memalign((void**)(&buffers[j]), 4096, buff_size);
	    for(int i=0; i<buff_size; i++) {
		buffers[j][i] = rank;
	    }
	}
	
	if(rank == 1) {
	    tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	/* send / recv niter messages, work in inflight requests at a time */
	int i = 0;
	while(i<niter){

	    for(int j=0; j<inflight; j++){
		if(rank==0 && (i)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);
		i++;
		ongoing_comm++;
		if(rank == 0)
		    comm.send(buffers[j], buff_size, 1, 42, send_callback);
		else
		    comm.recv(buffers[j], buff_size, 0, 42, recv_callback);
		if(i==niter) break;
	    }

	    /* complete all inflight requests before moving on */
	    while(ongoing_comm){
		comm.progress();
	    }
	}

	if(rank == 1) toc();
	comm.fence();
    }
}
