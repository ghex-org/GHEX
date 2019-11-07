#include <iostream>
#include <vector>
#include <omp.h>

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
#endif /* USE_MPI */

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt = 0, nlcomm_cnt = 0, submit_cnt = 0, submit_recv_cnt = 0;
int thrid, nthr;
#pragma omp threadprivate(comm_cnt, nlcomm_cnt, submit_cnt, submit_recv_cnt, thrid, nthr)

/* comm requests currently in-flight */
std::atomic<int> sent = 0;
std::atomic<int> received = 0;
int inflight;

void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    comm_cnt++;
    sent++;
}

void recv_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << " ongoing " << ongoing_comm << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    comm_cnt++;
    received++;
}

int main(int argc, char *argv[])
{
    int rank, size, threads, peer_rank;
    int niter, buff_size;

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
	gridtools::ghex::tl::callback_communicator<CommType> *comm 
	    = new gridtools::ghex::tl::callback_communicator<CommType>();

#pragma omp master
	{
	    rank = comm->rank();
	    size = comm->size();
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(*comm).name() << "\n\n";
	}

	std::vector<MsgType> smsgs;
	std::vector<MsgType> rmsgs;
	
	for(int j=0; j<inflight; j++){
	    smsgs.emplace_back(buff_size);
	    rmsgs.emplace_back(buff_size);
    	    make_zero(smsgs[j]);
    	    make_zero(rmsgs[j]);
	}
	
	comm->barrier();

#pragma omp master
	fprintf(stderr, "%d starting!\n", thrid);

#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size;
	}
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	/* send/recv niter messages - as soon as a slot becomes free */
	int i = 0, dbg = 0, rdbg = 0;
	fprintf(stderr, "%d starting loop!\n", thrid);
    	while(sent < niter || received < niter){
	    for(int j=0; j<inflight; j++){

		/* only send niter/nthr messages, with any tag */
		if(submit_cnt < niter && smsgs[j].use_count() == 1){
		    if(dbg >= (niter/10)) {
			std::cout << rank << ":" << thrid << "   " << submit_cnt << " sent dbg " << dbg << "\n";
			dbg = 0;
		    }
		    submit_cnt += nthr;
		    dbg += nthr;
		    comm->send(smsgs[j], peer_rank, thrid*inflight+j, send_callback);
		} else comm->progress();

		/* always submit a recv request for all tags - we don't know what will be sent */
		if(received < niter && rmsgs[j].use_count() == 1){
		    if(rdbg >= (niter/10)) {
			std::cout << rank << ":" << thrid << "   " << received << " received dbg " << rdbg << "\n";
			rdbg = 0;
		    }
		    rdbg += nthr;
		    comm->recv(rmsgs[j], peer_rank, thrid*inflight+j, recv_callback);
		} else comm->progress();
	    }
	}

	fprintf(stderr, "%d:%d runnig thread barrier!\n", rank, thrid);
	comm->barrier();

#pragma omp barrier

	fprintf(stderr, "%d:%d trying to end!\n", rank, thrid);

	comm->flush();
	comm->barrier();

#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " sends submitted " << submit_cnt/nthr
		  << " serviced " << comm_cnt << ", non-local " << nlcomm_cnt << " completion events\n";
	
	delete comm;
    }
    
    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    // MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
#endif
}
