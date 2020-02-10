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
#include <string.h>
#include <atomic>

#include <ghex/transport_layer/ucx/threads.hpp>
#include <ghex/common/timer.hpp>
#include "utils.hpp"

int main(int argc, char *argv[])
{
    int rank, size, peer_rank;
    int niter, buff_size;
    int inflight;

    gridtools::ghex::timer timer, ttimer;

    if(argc != 4){
	std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
	std::terminate();
    }
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);
    
    int mode;
#ifdef USE_OPENMP
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
#else
    MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &mode);
#endif

    THREAD_PARALLEL_BEG() {

	int thrid, nthr;
	MPI_Comm mpi_comm;
	unsigned char **sbuffers = new unsigned char *[inflight];
	unsigned char **rbuffers = new unsigned char *[inflight];
	MPI_Request *sreq = new MPI_Request[inflight];
	MPI_Request *rreq = new MPI_Request[inflight];
	
	THREAD_MASTER() {
	    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	    MPI_Comm_size(MPI_COMM_WORLD, &size);
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << "\n\n";
	}

	thrid = GET_THREAD_NUM();
	nthr = GET_NUM_THREADS();

	/* duplicate the communicator - all threads in order */
	for(int tid=0; tid<nthr; tid++){
	    if(thrid==tid) {
		MPI_Comm_dup(MPI_COMM_WORLD, &mpi_comm);
	    }
	    THREAD_BARRIER();
	}
	
	for(int j=0; j<inflight; j++){
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &sbuffers[j]);
	    MPI_Alloc_mem(buff_size, MPI_INFO_NULL, &rbuffers[j]);
	    memset(sbuffers[j], 1, buff_size);
	    memset(rbuffers[j], 1, buff_size);
	    sreq[j] = MPI_REQUEST_NULL;
	    rreq[j] = MPI_REQUEST_NULL;
	}

	THREAD_MASTER(){
	    MPI_Barrier(mpi_comm);
	}
	THREAD_BARRIER();

	THREAD_MASTER() {
	    timer.tic();
	    ttimer.tic();
	    if(rank == 1) std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
	}

	int i = 0, dbg = 0;
	int last_i = 0;
	char header[256];
	snprintf(header, 256, "%d total bwdt ", rank);
	while(i<niter){

	    if(thrid == 0 && dbg >= (niter/10)) {
	    	dbg = 0;
	    	timer.vtoc(header, (double)(i-last_i)*size*buff_size);
	    	timer.tic();
	    	last_i = i;
	    }

	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		MPI_Irecv(rbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &rreq[j]);
		MPI_Isend(sbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &sreq[j]);
		dbg +=nthr;
		i+=nthr;
	    }

	    /* wait for all to complete */
	    MPI_Waitall(inflight, sreq, MPI_STATUS_IGNORE);
	    MPI_Waitall(inflight, rreq, MPI_STATUS_IGNORE);
	}

	THREAD_MASTER(){
	    MPI_Barrier(mpi_comm);
	}
	THREAD_BARRIER();
    }

    if(rank == 1) {
	ttimer.vtoc();
	ttimer.vtoc("final ", (double)niter*size*buff_size);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
