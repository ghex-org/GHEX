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

	std::vector<MsgType> smsgs;
	std::vector<MsgType> rmsgs;
	FutureType sreqs[inflight];
	FutureType rreqs[inflight];
	
	for(int j=0; j<inflight; j++){
	    smsgs.push_back(MsgType(buff_size));
	    rmsgs.push_back(MsgType(buff_size));
	    make_zero(smsgs[j]);
	    make_zero(rmsgs[j]);
	}

	comm->barrier();

#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size;
	}

	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	int sent = 0, received = 0;
	int dbg = 0;	
	while(sent < niter || received < niter){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){

		if(sent < niter && sreqs[j].ready()) {
		    if(rank==0 && thrid==0 && dbg >= (niter/10)) {
			std::cout << sent << " iters\n";
			dbg = 0;
		    }
		    sent+=nthr;
		    dbg+=nthr;
		    sreqs[j] = comm->send(smsgs[j], peer_rank, j);
		}

		if(received < niter && rreqs[j].ready()) {		
		    received+=nthr;
		    rreqs[j] = comm->recv(rmsgs[j], peer_rank, j);
		}
	    }
	}

#pragma omp barrier
	comm->flush();
	comm->barrier();

	delete comm;
    }

    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize(); segfault ??
#endif
}
