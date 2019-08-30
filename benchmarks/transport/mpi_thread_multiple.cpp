#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

#define BUFF_SIZE 1024*1024
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
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threads);
    fprintf(stderr, "thread mode %d %d\n", MPI_THREAD_MULTIPLE, threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size!=2){
	fprintf(stderr, "ERROR: only works for 2 MPI ranks\n");
	exit(1);
    }
    peer_rank = (rank+1)%2;

#pragma omp parallel shared(niter, buff_size, peer_rank)
    {
	int thrid, nthr;
	unsigned char **sbuffer = new unsigned char *[inflight], **rbuffer = new unsigned char *[inflight];
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	fprintf(stderr, "rank %d thrid %d started\n", rank, thrid);
#pragma omp barrier

	for(int j=0; j<inflight; j++){
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &sbuffer[j]);
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &rbuffer[j]);
	
	    for(int i=0; i<buff_size; i++) {
		sbuffer[j][i] = i%(rank+thrid+1);
		rbuffer[j][i] = 0;
	    }
	}
	
#pragma omp master
	{
	    MPI_Barrier(MPI_COMM_WORLD);
	}
#pragma omp barrier
	fprintf(stderr, "rank %d thrid %d on omp barrier\n", rank, thrid);

#pragma omp master
	{
	    if(rank == 0) tic();
	    bytes = (double)niter*nthr*size*buff_size;
	}
	
	MPI_Request *sreq = new MPI_Request[inflight], *rreq = new MPI_Request[inflight];
	for(int i=0; i<niter; i+=inflight){

	    /* submitt inflight async requests, wait for all afterwards */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && (i+j)%10000==0) fprintf(stderr, "%d iters\n", i);
		MPI_Isend(sbuffer[j], buff_size, MPI_BYTE, peer_rank, j, MPI_COMM_WORLD, &sreq[j]);
		MPI_Irecv(rbuffer[j], buff_size, MPI_BYTE, peer_rank, j, MPI_COMM_WORLD, &rreq[j]);
	    }
	    MPI_Waitall(inflight, sreq, MPI_STATUS_IGNORE);
	    MPI_Waitall(inflight, rreq, MPI_STATUS_IGNORE);
	}

#pragma omp barrier
#pragma omp master
	{
	    MPI_Barrier(MPI_COMM_WORLD);
	    if(rank == 0) toc();
	}

	/* cleanup */
	for(int j=0; j<inflight; j++){
	    MPI_Free_mem(sbuffer[j]);
	    MPI_Free_mem(rbuffer[j]);
	}
	delete sreq;
	delete rreq;
	delete sbuffer;
	delete rbuffer;
    }
    
    MPI_Finalize();
}
