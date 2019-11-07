#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <string.h>

#include <ghex/common/timer.hpp>

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;
    MPI_Comm mpi_comm;
    int ncomm = 0;

    gridtools::ghex::timer timer;
    long bytes = 0;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &threads);
    MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
    MPI_Comm_rank(mpi_comm, &rank);
    MPI_Comm_size(mpi_comm, &size);
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << "\n\n";

#pragma omp parallel shared(niter, buff_size, peer_rank) reduction( + : ncomm )
    {
	int thrid, nthr;
	unsigned char **buffers = new unsigned char *[inflight];
	MPI_Request *req = new MPI_Request[inflight];
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	for(int j=0; j<inflight; j++){
	    req[j] = MPI_REQUEST_NULL;
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &buffers[j]);	
	    memset(buffers[j], 1, buff_size);
	}
	
#pragma omp master
	{
	    MPI_Barrier(MPI_COMM_WORLD);
	    if(rank == 1) {
		timer.tic();
		bytes = (double)niter*size*buff_size/2;
	    }
	}
#pragma omp barrier

	int i = 0, dbg = 0, flag, j;

	/* submit inflight async requests */
	for(j=0; j<inflight; j++){
	    if(rank==0)
		MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
	    else
		MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
	}
	
	while(i<niter){

	    /*
	      A version with a loop over inflight MPI_Test calls
	    */
	    /*
	    for(j=0; j<inflight; j++){

	    	MPI_Test(&req[j], &flag, MPI_STATUS_IGNORE);
	    	if(!flag) continue;
	    		
	    	if(rank==0) {
	    	    if(thrid == 0 && dbg>=(niter/10)) {
	    		std::cout << i << " iters\n";
	    		dbg = 0;
	    	    }		    
	    	    MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
	    	} else
	    	    MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);

	    	ncomm++;
	    	dbg +=nthr; i+=nthr;
	    }
	    */	    

	    /* A version with MPI_Testany instead of an explicit loop : both are the same */
	    MPI_Testany(inflight, req, &j, &flag, MPI_STATUS_IGNORE);	    
	    if(flag) {
	    	if(rank==0){
	    	    if(thrid==0 && dbg>=(niter/10)) {
	    		std::cout << i << " iters\n";
	    		dbg=0;
	    	    }
	    	    MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);
	    	} else
	    	    MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &req[j]);

	    	ncomm++;
	    	dbg +=nthr; i+=nthr;
	    }
	}
	std::cout << "rank " << rank << " thrid " << thrid << " ncomm " << ncomm << "\n";

#pragma omp barrier
#pragma omp master
	{
	    MPI_Barrier(MPI_COMM_WORLD);
	    if(rank == 1) timer.vtoc(bytes);
	}
#pragma omp barrier
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
