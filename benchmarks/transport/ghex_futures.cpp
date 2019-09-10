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


int main(int argc, char *argv[])
{
    gridtools::ghex::ucx::communicator::rank_type rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    pmi_init();

    // MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threads);

    rank = pmi_get_rank();
    size = pmi_get_size();
    peer_rank = (rank+1)%2;
    printf("PMIx rank size %d %d\n", rank, size);
    
    // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    // MPI_Comm_size(MPI_COMM_WORLD, &size);
    // printf("MPI rank size %d %d\n", rank, size);    
    gridtools::ghex::ucx::communicator comm;


#pragma omp parallel shared(inflight, comm)
    {
	int thrid, nthr;

	using MsgType = gridtools::ghex::mpi::message<>;
	std::vector<MsgType> msgs;
	gridtools::ghex::ucx::communicator::future_type reqs[inflight];	
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);

#pragma omp critical
	for(int j=0; j<inflight; j++){
	    msgs.push_back(MsgType(buff_size, buff_size));
	}
	fprintf(stderr, "messages created\n");

#pragma omp barrier
#pragma omp master
	if(rank == 0) {
	    tic();
	    bytes = (double)niter*nthr*size*buff_size/2;
	}

	for(int i=0; i<niter; i+=inflight){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && (i+j)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);

		fprintf(stderr, "comm start\n");
		if(rank == 0)
		    reqs[j] = comm.send(msgs[j], 1, 42);
		else
		    reqs[j] = comm.recv(msgs[j], 0, 42);
		fprintf(stderr, "comm end\n");
	    }

	    /* wait for comm */
	    for(int j=0; j<inflight; j++){
		reqs[j].wait();
	    }
	}

#pragma omp master
	if(rank == 0) toc();    

    }


    pmi_finalize();
    // MPI_Finalize();
}
