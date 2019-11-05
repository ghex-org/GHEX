#include <iostream>
#include <mpi.h>
#include <string.h>

#include <ghex/common/timer.hpp>

int main(int argc, char *argv[])
{
    int rank, size, mode, peer_rank;
    int niter, buff_size;
    int inflight;
    MPI_Comm mpi_comm;

    gridtools::ghex::timer timer;
    long bytes = 0;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
#ifdef THREAD_MODE_MULTIPLE
	MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
#else
	MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif

    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << "\n\n";

    {
	unsigned char **sbuffers = new unsigned char *[inflight];
	unsigned char **rbuffers = new unsigned char *[inflight];
	MPI_Request *sreq = new MPI_Request[inflight];
	MPI_Request *rreq = new MPI_Request[inflight];
	
	for(int j=0; j<inflight; j++){
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &sbuffers[j]);
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &rbuffers[j]);
	    memset(sbuffers[j], 1, buff_size);
	    memset(rbuffers[j], 1, buff_size);
	    sreq[j] = MPI_REQUEST_NULL;
	    rreq[j] = MPI_REQUEST_NULL;
	}
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size;
	}

	int sent = 0;
	int received = 0;
	while(sent < niter || received < niter){

	    if(rank==0 && sent%(niter/10)==0) {
		std::cout << sent << " iters\n";
	    }
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		MPI_Isend(sbuffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &sreq[j]);
		MPI_Irecv(rbuffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &rreq[j]);
	    }

	    /* wait for all */
	    MPI_Waitall(inflight, sreq, MPI_STATUS_IGNORE);
	    MPI_Waitall(inflight, rreq, MPI_STATUS_IGNORE);
	    sent+=inflight;
	    received+=inflight;
	    
	    // for(int j=0; j<inflight; j++){
	    // 	MPI_Wait(&sreq[j], MPI_STATUS_IGNORE);
	    // 	MPI_Wait(&rreq[j], MPI_STATUS_IGNORE);			
	    // 	sent++;
	    // 	received++;
	    // }
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 1) timer.vtoc(bytes);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
}
