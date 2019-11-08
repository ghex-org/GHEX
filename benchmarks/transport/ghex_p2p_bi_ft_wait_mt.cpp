#include <iostream>
#include <vector>

#include <ghex/transport_layer/ucx/threads.hpp>
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
DECLARE_THREAD_PRIVATE(thrid)
DECLARE_THREAD_PRIVATE(nthr)

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

    THREAD_PARALLEL_BEG() {
	CommType comm;

	THREAD_MASTER() {
	    rank = comm.rank();
	    size = comm.size();
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";
	}

	thrid = GET_THREAD_NUM();
	nthr = GET_NUM_THREADS();

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

	comm.barrier();

	THREAD_MASTER() {
	    if(rank == 1) {
		std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
		timer.tic();
		bytes = (double)niter*size*buff_size;
	    }
	}

	int dbg = 0;	
	int sent = 0, received = 0;
	while(sent < niter || received < niter){
	    	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		
		if(rank==0 && thrid==0 && dbg>=(niter/10)) {
		    std::cout << sent << " iters\n";
		    dbg=0;
		}
		dbg += nthr; 
		sent += nthr;
		received += nthr;

		rreqs[j] = comm.recv(rmsgs[j], peer_rank, thrid*inflight + j);
		sreqs[j] = comm.send(smsgs[j], peer_rank, thrid*inflight + j);
	    }

	    /* wait for all */
	    for(int j=0; j<inflight; j++){
		sreqs[j].wait();
		rreqs[j].wait();		
	    }
	}

	comm.barrier();
	comm.finalize();
	
    } THREAD_PARALLEL_END();

    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize(); segfault ??
#endif
}
