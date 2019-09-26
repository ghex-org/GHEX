#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>
#include <iostream>
#include <vector>
#include <array>
#include <unistd.h>
#include <sched.h>
#include <mpi.h>
#include <omp.h>

#include "debug.h"
#include "tictoc.h"
#include "pmi.h"

#ifdef USE_MPI
#include "communicator_mpi.hpp"
using CommType = gridtools::ghex::mpi::communicator;
using FutureType = gridtools::ghex::mpi::communicator::future_type;
using namespace gridtools::ghex::mpi;
#else
#include "communicator_ucx.hpp"
using CommType = gridtools::ghex::ucx::communicator;
using FutureType = gridtools::ghex::ucx::communicator::future_type;
using namespace gridtools::ghex::ucx;
#endif

#include "message.hpp"
using MsgType = gridtools::ghex::mpi::shared_message<>;

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << comm.name << "\n\n";
    
    {
	std::vector<MsgType> msgs;
	FutureType reqs[inflight];
	
	for(int j=0; j<inflight; j++){
	    msgs.push_back(MsgType(buff_size, buff_size));
	}

	if(rank == 1) {
	    tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	int i = 0;
	while(i<niter){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){

		if(!reqs[j].ready()) continue;

		i++;
		if(rank==0 && (i)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);

		if(rank == 0)
		    reqs[j] = comm.send(msgs[j], 1, j);
		else
		    reqs[j] = comm.recv(msgs[j], 0, j);
		if(i==niter) break;
	    }
	}

	if(rank == 1) toc();

	for(int j=0; j<inflight; j++){
	    reqs[i].cancel();
	}

	comm.fence();
    }
}
