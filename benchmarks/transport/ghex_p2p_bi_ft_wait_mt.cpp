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

/* use internal UCX locks instead of
   having  GHEX lock over the recv worker.
   That works only in pure future-based code.
*/
// #ifdef USE_OPENMP
// #undef THREAD_MODE_SERIALIZED
// #define THREAD_MODE_MULTIPLE
// #endif

#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
using FutureType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>::request;

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
    gridtools::ghex::timer timer, ttimer;

#ifdef USE_MPI
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
#endif
    
    if(argc != 4){
	std::cerr << "Usage: bench [niter] [msg_size] [inflight]" << "\n";
	std::terminate();
    }
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
	FutureType *sreqs = new FutureType[inflight];
	FutureType *rreqs = new FutureType[inflight];
	
	for(int j=0; j<inflight; j++){
	    smsgs.push_back(MsgType(buff_size));
	    rmsgs.push_back(MsgType(buff_size));
	    make_zero(smsgs[j]);
	    make_zero(rmsgs[j]);
	}

	comm.barrier();

	THREAD_MASTER() {
	    timer.tic();
	    ttimer.tic();
	    if(rank == 1) std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
	}

	int dbg = 0;
	int sent = 0, received = 0;
	int last_received = 0;
	int last_sent = 0;
	char header[256];
	snprintf(header, 256, "%d total bwdt ", rank);
	while(sent < niter || received < niter){
	    	    
	    if(thrid == 0 && dbg >= (niter/10)) {
		dbg = 0;
		timer.vtoc(header, (double)(received-last_received + sent-last_sent)*size*buff_size/2);
		timer.tic();
		last_received = received;
		last_sent = sent;
	    }
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){
		
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

	THREAD_MASTER() {
	    if(rank == 1) {
		ttimer.vtoc();
		ttimer.vtoc("final ", (double)niter*size*buff_size);
	    }
	}

	/* tail loops - not needed in wait benchmarks */
	
    } THREAD_PARALLEL_END();

    CommType::finalize();

#ifdef USE_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize(); segfault ??
#endif
}
