#include <iostream>
#include <mpi.h>
#include "tictoc.h"

int main(int argc, char *argv[])
{
    int rank, size, mode, peer_rank;
    int niter, buff_size;
    int inflight;
    MPI_Comm mpi_comm;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
#ifdef THREAD_MODE_MULTIPLE
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
#else
	// MPI_Init(NULL, NULL);
	MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << "\n\n";

    {
	unsigned char **buffers = new unsigned char *[inflight];
	MPI_Request *req = new MPI_Request[inflight];
	
	for(int j=0; j<inflight; j++){
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &buffers[j]);
	    req[j] = MPI_REQUEST_NULL;
	    for(int i=0; i<buff_size; i++) {
		buffers[j][i] = i%(rank+1);
	    }
	}
	
	if(rank == 1) {
	    tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	/* submit inflight async requests */
	for(int j=0; j<inflight; j++){
	    if(rank==0)
		MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
	    else
		MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
	}
	
	int i = 0;
	while(i<niter){
	    for(int j=0; j<inflight; j++){

		int flag;
		MPI_Test(&req[j], &flag, MPI_STATUS_IGNORE);
		if(!flag) continue;
	    
		if(rank==0 && i%(niter/10)==0) {
		    std::cout << i << " iters\n";
		}
		
		if(rank==0)
		    MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
		else
		    MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
		i++; if(i==niter) break;		
	    }
	}

	if(rank == 1) toc();
    }    
    MPI_Finalize();
}
