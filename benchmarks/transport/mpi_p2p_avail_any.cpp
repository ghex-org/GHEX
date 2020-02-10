/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#include <iostream>
#include <mpi.h>

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
    
#ifdef USE_OPENMP
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
	
	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}
	
	/* submit inflight async requests */
	for(int j=0; j<inflight; j++){
	    if(rank==0)
		MPI_Isend(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
	    else
		MPI_Irecv(buffers[j], buff_size, MPI_BYTE, peer_rank, j, mpi_comm, &req[j]);
	}

	int i = 0, j, dbg = 0, thrid = 0, nthr = 1;
	while(i<niter){
	    int completed, flag;


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

	    	dbg +=nthr; i+=nthr;
	    }

	    // MPI_Waitany(inflight, req, &completed, MPI_STATUS_IGNORE);
	    // // MPI_Testany(inflight, req, &completed, &flag, MPI_STATUS_IGNORE);
	    // // if(!flag) continue;
	    
	    // if(rank==0 && i%(niter/10)==0) {
	    // 	std::cout << i << " iters\n";
	    // }

	    // if(rank==0)
	    // 	MPI_Isend(buffers[completed], buff_size, MPI_BYTE, peer_rank, completed, mpi_comm, &req[completed]);
	    // else
	    // 	MPI_Irecv(buffers[completed], buff_size, MPI_BYTE, peer_rank, completed, mpi_comm, &req[completed]);	    
	    // i++; if(i==niter) break;
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(rank == 1) timer.vtoc(bytes);
    }
    
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
