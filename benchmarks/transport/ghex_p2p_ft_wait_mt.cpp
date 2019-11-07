#include <iostream>
#include <vector>
#include <omp.h>

#include <ghex/common/timer.hpp>
#include "utils.hpp"

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


int thrid, nthr;
#pragma omp threadprivate(thrid, nthr)

int main(int argc, char *argv[])
{
    int rank, size, peer_rank;
    int niter, buff_size;
    int inflight;
    gridtools::ghex::timer timer;
    long bytes = 0;

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
    
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   

#pragma omp parallel
    {
	CommType *comm = new CommType();

#pragma omp master
	{
	    rank = comm->rank();
	    size = comm->size();
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(*comm).name() << "\n\n";
	}
#pragma omp barrier 

	std::vector<MsgType> msgs;
	FutureType reqs[inflight];

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size);
	    make_zero(msgs[j]);
	}

	comm->barrier();

#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	int i = 0, dbg = 0;	
	while(i<niter){
	    	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		
		if(rank==0 && thrid==0 && dbg>=(niter/10)) {
		    std::cout << i << " iters\n";
		    dbg=0;
		}
		i += nthr;
		dbg += nthr; 

		if(rank == 0)
		    reqs[j] = comm->send(msgs[j], peer_rank, j);
		else
		    reqs[j] = comm->recv(msgs[j], peer_rank, j);
	    }

	    /* wait for all */
	    for(int j=0; j<inflight; j++){
		reqs[j].wait();
	    }
	}

#pragma omp barrier
	comm->flush();
	comm->barrier();

	delete comm;
    }

    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize(); segfault ??
#endif
}
