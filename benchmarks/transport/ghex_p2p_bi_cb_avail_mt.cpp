#include <iostream>
#include <vector>
#include <atomic>

#include <ghex/transport_layer/ucx/threads.hpp>
#include <ghex/common/timer.hpp>
#include "utils.hpp"


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
#else

/* UCX backend */
#ifdef USE_UCX_NBR
#include <ghex/transport_layer/callback_communicator.hpp>
#else
#include <ghex/transport_layer/ucx/callback_communicator.hpp>
#endif
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
using FutureType = gridtools::ghex::tl::callback_communicator<CommType>::request;
#endif /* USE_MPI */

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


/* comm requests currently in-flight */
std::atomic<int> sent(0);
std::atomic<int> received(0);
int last_received = 0;
int last_sent = 0;

int inflight;

/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt = 0, nlsend_cnt = 0, nlrecv_cnt = 0, submit_cnt = 0, submit_recv_cnt = 0;
DECLARE_THREAD_PRIVATE(comm_cnt)
DECLARE_THREAD_PRIVATE(nlsend_cnt)
DECLARE_THREAD_PRIVATE(nlrecv_cnt)
DECLARE_THREAD_PRIVATE(submit_cnt)
DECLARE_THREAD_PRIVATE(submit_recv_cnt)

int thrid, nthr;
DECLARE_THREAD_PRIVATE(thrid)
DECLARE_THREAD_PRIVATE(nthr)


void send_callback(MsgType, int, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlsend_cnt++;
    comm_cnt++;
    sent++;
}

void recv_callback(MsgType, int, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << " ongoing " << ongoing_comm << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlrecv_cnt++;
    comm_cnt++;
    received++;
}

int main(int argc, char *argv[])
{
    int rank, size, peer_rank;
    int niter, buff_size;

    gridtools::ghex::timer timer, ttimer;

    /* has to be done before MPI_Init */
    CommType::initialize();

    int mode;
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
	gridtools::ghex::tl::callback_communicator<CommType> comm;

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
	std::vector<FutureType> sreqs;
	std::vector<FutureType> rreqs;
	
	for(int j=0; j<inflight; j++){
	    smsgs.emplace_back(buff_size);
	    rmsgs.emplace_back(buff_size);
    	    make_zero(smsgs[j]);
    	    make_zero(rmsgs[j]);
	}

	sreqs.resize(inflight);
	rreqs.resize(inflight);	
	
	THREAD_MASTER() {
	    MPI_Barrier(MPI_COMM_WORLD);
	    timer.tic();
	    ttimer.tic();
	    if(rank == 1) std::cout << "number of threads: " << nthr << ", multi-threaded: " << THREAD_IS_MT << "\n";
	}
	THREAD_BARRIER();
	
	/* send/recv niter messages - as soon as a slot becomes free */
	int dbg = 0, sdbg = 0, rdbg = 0;
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
	    if(rank==0 && thrid==0 && rdbg >= (niter/10)) {
		std::cout << received << " received\n";
		rdbg = 0;
	    }

	    if(rank==0 && thrid==0 && sdbg >= (niter/10)) {
		std::cout << sent << " sent\n";
		sdbg = 0;
	    }

	    for(int j=0; j<inflight; j++){
		if(rmsgs[j].use_count() == 1){
		    submit_recv_cnt += nthr;
		    rdbg += nthr;
		    dbg += nthr;
		    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thrid*inflight+j, recv_callback);
		} else comm.progress();

		if(sent < niter && smsgs[j].use_count() == 1){
		    submit_cnt += nthr;
		    sdbg += nthr;
		    dbg += nthr;
		    sreqs[j] = comm.send(smsgs[j], peer_rank, thrid*inflight+j, send_callback);
		} else comm.progress();
	    }
	}

	THREAD_BARRIER()
	THREAD_MASTER() {
	    MPI_Barrier(MPI_COMM_WORLD);
	    if(rank == 1) {
		ttimer.vtoc();
		ttimer.vtoc("final ", (double)niter*size*buff_size);
	    }
	}
	THREAD_BARRIER()

#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " sends submitted " << submit_cnt/nthr
		  << " serviced " << comm_cnt << ", non-local sends " << nlsend_cnt << " non-local recvs " << nlrecv_cnt << "\n";

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
		if(rmsgs[j].use_count() == 1){
		    rreqs[j] = comm.recv(rmsgs[j], peer_rank, thrid*inflight+j, recv_callback);
		}
	    }
	} while(incomplete_sends);

	/* this will make sure everyone has progressed all sends... */
	THREAD_BARRIER()
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
