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

#include <ghex/common/timer.hpp>

#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
using FutureType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>::future<void>;
#else

/* UCX backend */
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
using FutureType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>::future<void>;
#endif /* USE_MPI */


#include <ghex/transport_layer/message_buffer.hpp>
using MsgType = gridtools::ghex::tl::message_buffer<>;


int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;
    int inflight;
    
#ifdef USE_MPI
    int mode;
#ifdef THREAD_MODE_MULTIPLE
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
#endif
#endif

    /* TODO this needs to be made per-thread. 
       If we make 'static' variables, then we can't initialize m_rank and anything else
       that used MPI in the constructor, as it will be executed before MPI_Init.
    */
    CommType comm;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);

    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
    
    {
	gridtools::ghex::timer timer;
	long bytes = 0;
	std::vector<MsgType> msgs;
	FutureType reqs[inflight];
	
	for(int j=0; j<inflight; j++){
	    msgs.push_back(MsgType(buff_size));
	}

	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	int i = 0;
	while(i<niter){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){

		if(!reqs[j].ready()) continue;

		i++;
		if(rank==0 && (i)%(niter/10)==0) {
		    std::cout << i << " iters\n";		    
		}

		if(rank == 0)
		    reqs[j] = comm.send(1, j, msgs[j]);
		else
		    reqs[j] = comm.recv(0, j, msgs[j]);
		if(i==niter) break;
	    }
	}

	if(rank == 1) timer.vtoc(bytes);
	// comm.fence();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
}
