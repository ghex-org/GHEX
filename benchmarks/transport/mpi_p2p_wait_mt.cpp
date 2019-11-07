#include <iostream>
#include <mpi.h>
#include <string.h>
#include <omp.h>

#include <ghex/common/timer.hpp>

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;
    int ncomm = 0;

    gridtools::ghex::timer timer;
    long bytes = 0;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threads);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << "\n\n";

#pragma omp parallel shared(niter, buff_size, peer_rank) reduction( + : ncomm )
    {
	int thrid, nthr;
	unsigned char **buffers = new unsigned char *[inflight];
	MPI_Request *req = new MPI_Request[inflight];
	MPI_Comm mpi_comm;
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	/* duplicate the communicator - all threads in order */
	for(int tid=0; tid<nthr; tid++){
	    if(thrid==tid) {
		MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
	    }
#pragma omp barrier
	}
	MPI_Barrier(mpi_comm);

	std::cout << "rank " << rank << " thrid "<< thrid << " started\n";

	for(int j=0; j<inflight; j++){
	    req[j] = MPI_REQUEST_NULL;
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &buffers[j]);	
	    memset(buffers[j], 1, buff_size);
	}
	
	MPI_Barrier(mpi_comm);
#pragma omp barrier

#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*buff_size;
	}

	int i = 0, dbg = 0;
	while(i<niter){

	    /* submit inflight async requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && dbg>=(niter/10)) {
		    std::cout << i << " iters\n";
		    dbg=0;
		}
		if(rank==0)
		    MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
		else
		    MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
		ncomm++;
		dbg +=nthr; i+=nthr; if(i==niter) break;
	    }

	    /* wait for all to complete */
	    MPI_Waitall(inflight, req, MPI_STATUS_IGNORE);
	}

	MPI_Barrier(mpi_comm);
    }

    if(rank == 1) timer.vtoc(bytes);

    std::cout << "rank " << rank << " ncomm " << ncomm << "\n";
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
