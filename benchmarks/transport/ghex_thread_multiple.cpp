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
#include "persistent_allocator.hpp"
#include "communicator.hpp"
#include "message.hpp"

#include <mpi.h>
#include <omp.h>

static struct timeval tb, te;
double bytes = 0;

#define RAW_POINTERS

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


/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int ongoing_comm = 0;

/* per-thread allocator */
using AllocType = ghex::allocator::persistent_allocator<unsigned char, std::allocator<unsigned char>>;
AllocType allocator;
int buff_size;

int comm_cnt;
int submit_cnt;
#pragma omp threadprivate(comm_cnt, submit_cnt)

void send_callback(int rank, int tag, void *mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
#pragma omp atomic
    ongoing_comm--;

    comm_cnt++;
#ifdef RAW_POINTERS
    allocator.deallocate((unsigned char *)mesg, buff_size);
#endif
}

void recv_callback(int rank, int tag, void *mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
#pragma omp atomic
    ongoing_comm--;

    comm_cnt++;
#ifdef RAW_POINTERS
    allocator.deallocate((unsigned char *)mesg, buff_size);
#endif
}

int main(int argc, char *argv[])
{
    gridtools::ghex::ucx::communicator::rank_type rank, size, threads, peer_rank;
    int niter;
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
	using AllocType = ghex::allocator::persistent_allocator<unsigned char, std::allocator<unsigned char>>;
	using MsgType = gridtools::ghex::mpi::shared_message<AllocType>;
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);

#pragma omp barrier
#pragma omp master
	if(rank == 0) {
	    tic();
	    bytes = (double)niter*nthr*size*buff_size/2;
	}

	comm_cnt = 0;
	submit_cnt = 0;
	int i = 0;
	while(i<niter){
	    while(i<niter && ongoing_comm < inflight){
		if(rank==0 && thrid==0 && (i)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);

		i++;
		submit_cnt++;
#pragma omp atomic
		ongoing_comm++;

#ifdef RAW_POINTERS
		unsigned char *buffer = allocator.allocate(buff_size);
		if(rank == 0)
		    comm.send(buffer, buff_size, 1, 42, send_callback);
		else
		    comm.recv(buffer, buff_size, 0, 42, recv_callback);
#else
		MsgType msg(buff_size, buff_size);
		if(rank == 0)
		    comm.send(msgs[j], 1, 42, send_callback);
		else
		    comm.recv(msgs[j], 0, 42, recv_callback);
#endif
	    }

	    /* always progress */
	    do {
		comm.progress();
	    } while(ongoing_comm);
	}

#pragma omp master
	if(rank == 0) toc();    

#pragma omp barrier
	printf("rank %d thread %d submitted %d serviced %d completion events\n", rank, thrid, submit_cnt, comm_cnt);
    }
    
    pmi_finalize();
    // MPI_Finalize();
}
