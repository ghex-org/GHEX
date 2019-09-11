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
CommType comm;

/* available request slots */
int *available = NULL;
int ongoing_comm = 0;

void send_callback(int rank, int tag, void *mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;    
}

void recv_callback(int rank, int tag, void *mesg)
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

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << comm.name << "\n\n";    
    
    {
	unsigned char **buffers = new unsigned char *[inflight];
	available = new int[inflight];

	for(int j=0; j<inflight; j++){
	    available[j] = 1;
	    posix_memalign((void**)(&buffers[j]), 4096, buff_size);
	    for(int i=0; i<buff_size; i++) {
		buffers[j][i] = rank;
	    }
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
			comm.send(buffers[j], buff_size, 1, j, send_callback);
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
			comm.recv(buffers[j], buff_size, 0, j, recv_callback);		    
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
	comm.fence();
    }
}
