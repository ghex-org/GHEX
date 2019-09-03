#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>

#include "pmi.h"
#include "communicator.hpp"
#include "message.hpp"

#include <mpi.h>
#include <omp.h>

static struct timeval tb, te;
double bytes = 0;

void tic(void)
{ 
#pragma omp master
  {
    gettimeofday(&tb, NULL);
    bytes = 0;
  }
}

void toc(void)
{ 
#pragma omp master
  {
    long s,u;
    double tt;
    gettimeofday(&te, NULL);
    s=te.tv_sec-tb.tv_sec;
    u=te.tv_usec-tb.tv_usec;
    tt=((double)s)*1000000+u;
    fprintf(stderr, "time:                  %li.%.6lis\n", (s*1000000+u)/1000000, (s*1000000+u)%1000000);
    fprintf(stderr, "MB/s:                  %.3lf\n", bytes/tt);
  }
}


int ready_comm = 0;
void send_callback(int rank, int tag, void *mesg)
{
    //std::cout << "send callback called " << rank << " tag " << tag << "\n";
    ready_comm++;
}

void recv_callback(int rank, int tag, void *mesg)
{
    //std::cout << "recv callback called " << rank << " tag " << tag << "\n";
    ready_comm++;
}

int main(int argc, char *argv[])
{
    gridtools::ghex::ucx::communicator::rank_type rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threads);
    pmi_init();

    rank = pmi_get_rank();
    size = pmi_get_size();
    peer_rank = (rank+1)%2;

    gridtools::ghex::ucx::communicator comm;

#pragma omp parallel
    {
	int thrid, nthr;
	unsigned char **buffers = new unsigned char *[inflight];
	std::vector<gridtools::ghex::mpi::shared_message<>> msgs[inflight];
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);

	for(int j=0; j<inflight; j++){
	    posix_memalign((void**)(&buffers[j]), 4096, buff_size);
	    for(int i=0; i<buff_size; i++) {
	    	buffers[j][i] = i%(rank+thrid+1);
	    }
	    
	    msgs[j] = gridtools::ghex::mpi::shared_message<>(buff_size, buff_size);
	}

	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();
	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);

#pragma omp master
	if(rank == 0) {
	    tic();
	    bytes = (double)niter*nthr*size*buff_size/2;
	}

	for(int i=0; i<niter; i+=inflight){

	    ready_comm = 0;
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && (i+j)%10000==0) fprintf(stderr, "%d iters\n", i);
		if(rank == 0)
		    comm.send(msgs[j], 1, 42, send_callback);
		else
		    comm.recv(msgs[j], 0, 42, recv_callback);
		// if(rank == 0)
		//     comm.send(buffers[j], buff_size, 1, 42, send_callback);
		// else
		//     comm.recv(buffers[j], buff_size, 0, 42, recv_callback);
	    }

	    while(ready_comm != inflight){
		comm.progress();
	    }
	}

#pragma omp master
	if(rank == 0) toc();    }


    // MPI_Finalize();
    pmi_finalize();
}
