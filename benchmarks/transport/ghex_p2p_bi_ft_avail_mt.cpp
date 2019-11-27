#include <iostream>
#include <vector>
#include <atomic>

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


std::atomic<int> sent(0);
std::atomic<int> received(0);
int last_received = 0;
int last_sent = 0;

int thrid, nthr;
DECLARE_THREAD_PRIVATE(thrid)
DECLARE_THREAD_PRIVATE(nthr)

int main(int argc, char *argv[])
{
    int rank, size, peer_rank;
    int niter, buff_size;
    int inflight;
    int mode;
    gridtools::ghex::timer timer, ttimer;

    /* has to be done before MPI_Init */
    CommType::initialize();    

#ifdef USE_OPENMP
    MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &mode);
    if(mode != MPI_THREAD_MULTIPLE){
	std::cerr << "MPI_THREAD_MULTIPLE not supported by MPI, aborting\n";
	std::terminate();
    }
#else
    MPI_Init_thread(NULL, NULL, MPI_THREAD_SINGLE, &mode);
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

	THREAD_MASTER() {
	    MPI_Barrier(MPI_COMM_WORLD);
	    timer.tic();
	    ttimer.tic();
	    if(rank == 1) std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
	}
	THREAD_BARRIER();

	int dbg = 0, sdbg = 0, rdbg = 0;
	char header[256];
	snprintf(header, 256, "%d total bwdt ", rank);
	while(sent < niter || received < niter){
	    
	    /* submit comm */
	    for(int j=0; j<inflight; j++){

		if(rank==0 && thrid==0 && sdbg>=(niter/10)) {
		    std::cout << sent << " sent\n";
		    sdbg = 0;
		}

		if(rank==0 && thrid==0 && rdbg>=(niter/10)) {
		    std::cout << received << " received\n";
		    rdbg = 0;
		}

		if(thrid == 0 && dbg >= (niter/10)) {
		    dbg = 0;
		    timer.vtoc(header, (double)(received-last_received + sent-last_sent)*size*buff_size/2);
		    timer.tic();
		    last_received = received;
		    last_sent = sent;
		}

		if(rreqs[j].test()) {
		    received++;
		    rdbg+=nthr;
		    dbg+=nthr;
		    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thrid*inflight + j);
		}

		if(sent < niter && sreqs[j].test()) {
		    sent++;
		    sdbg+=nthr;
		    dbg+=nthr;
		    sreqs[j] = comm.send(smsgs[j], peer_rank, thrid*inflight + j);
		}
	    }
	}

	THREAD_BARRIER();
	THREAD_MASTER() {
	    MPI_Barrier(MPI_COMM_WORLD);
	    if(rank == 1) {
		ttimer.vtoc();
		ttimer.vtoc("final ", (double)niter*size*buff_size);
	    }
	}

	/* tail loops - submit RECV requests until
	   all SEND requests have been finalized.
	   This is because UCX cannot cancel SEND requests.
	   https://github.com/openucx/ucx/issues/1162
	*/
	int incomplete_sends = 0;
	do {
	    comm.progress();

	    incomplete_sends = 0;
	    for(int j=0; j<inflight; j++){
		if(!sreqs[j].test()) incomplete_sends++;
	    }

	    for(int j=0; j<inflight; j++){
		if(rreqs[j].test()) {
		    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thrid*inflight + j);
		}
	    }
	} while(incomplete_sends);

	/* this will make sure everyone has progressed all sends... */
	THREAD_BARRIER();
	THREAD_MASTER() {
	    MPI_Barrier(MPI_COMM_WORLD);
	}

	/* ... so we can cancel all RECV requests */
	for(int j=0; j<inflight; j++){
	    rreqs[j].cancel();
	}

    } THREAD_PARALLEL_END();

    CommType::finalize();

    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
}
