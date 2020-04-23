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

#include <ghex/common/timer.hpp>
#include "utils.hpp"

#ifdef USE_OPENMP
#include <omp.h>
#define DO_PRAGMA(x) _Pragma(#x)

#define GET_THREAD_NUM()  omp_get_thread_num()
#define GET_NUM_THREADS() omp_get_num_threads()
#define IN_PARALLEL()     omp_in_parallel()

#define THREAD_BARRIER()      DO_PRAGMA(omp barrier)
#define THREAD_MASTER()       DO_PRAGMA(omp master)
#define THREAD_PARALLEL_BEG() DO_PRAGMA(omp parallel)
#define THREAD_PARALLEL_END() 
#define THREAD_IS_MT 1

#else

#define GET_THREAD_NUM()  0
#define GET_NUM_THREADS() 1
#define IN_PARALLEL()     0

#define DECLARE_THREAD_PRIVATE(name)
#define THREAD_BARRIER()              
#define THREAD_MASTER()               
#define THREAD_PARALLEL_BEG()
#define THREAD_PARALLEL_END() 
#define THREAD_IS_MT 0

#endif /* USE_OPENMP */

std::atomic<int> sent(0);
std::atomic<int> received(0);
int last_received = 0;
int last_sent = 0;

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

	/* pre-post - required for testany to work */
        for(int j=0; j<inflight; j++){
            MPI_Irecv(rbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &rreq[j]);
            MPI_Isend(sbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &sreq[j]);
        }

        int dbg = 0, sdbg = 0, rdbg = 0, flag, j;
	char header[256];
	snprintf(header, 256, "%d total bwdt ", rank);
	while(sent<niter || received<niter){

	    if(rank==0 && thrid==0 && sdbg>=(niter/10)) {
		std::cout << sent << " sent\n";
		sdbg = 0;
	    }

	    if(rank==0 && thrid==0 && rdbg>=(niter/10)) {
		std::cout << received << " received\n";
		rdbg = 0;
	    }

	    if(thrid == 0 && dbg >= (2*niter/10)) {
	    	dbg = 0;
	    	timer.vtoc(header, (double)(received-last_received + sent-last_sent)*size*buff_size/2);
	    	timer.tic();
	    	last_received = received;
	    	last_sent = sent;
	    }

            // testany version is much faster with OpenMPI, esp. for large messages 
#define USE_TESTANY
#ifdef USE_TESTANY
	    MPI_Testany(inflight, rreq, &j, &flag, MPI_STATUS_IGNORE);
	    if(flag) {
	        MPI_Irecv(rbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &rreq[j]);
	        dbg += nthr;
	        rdbg += nthr;
	        received++;
	    }

	    if(sent<niter){
	        MPI_Testany(inflight, sreq, &j, &flag, MPI_STATUS_IGNORE);
	        if(flag) {
	            MPI_Isend(sbuffers[j], buff_size, MPI_BYTE, peer_rank, thrid*inflight+j, mpi_comm, &sreq[j]);
	            dbg += nthr;
	            sdbg += nthr;
	            sent++;
	        }
	    }
#else
            for(int i=0; i<inflight; i++){
                MPI_Test(&rreq[i], &j, MPI_STATUS_IGNORE);
                if(j){
                    MPI_Irecv(rbuffers[i], buff_size, MPI_BYTE, peer_rank, thrid*inflight+i, mpi_comm, &rreq[i]);
                    dbg += nthr;
                    rdbg += nthr;
                    received++;
                }
            }
            for(int i=0; i<inflight; i++){
                MPI_Test(&sreq[i], &j, MPI_STATUS_IGNORE);
                if(j){
                    MPI_Isend(sbuffers[i], buff_size, MPI_BYTE, peer_rank, thrid*inflight+i, mpi_comm, &sreq[i]);
                    dbg += nthr;
                    sdbg += nthr;
                    sent++;
                }
            }
#endif
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
    // MPI_Finalize(); // segfault due to unfinished requests
}
