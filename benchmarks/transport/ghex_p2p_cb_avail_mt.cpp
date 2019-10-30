#include <iostream>
#include <vector>
#include <omp.h>

#include <ghex/common/timer.hpp>


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/callback_communicator.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
#else

/* UCX backend */
#include <ghex/transport_layer/ucx/callback_communicator.hpp>
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;
#endif /* USE_MPI */

using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt, nlcomm_cnt;
int submit_cnt;
int thrid, nthr;
#pragma omp threadprivate(comm_cnt, nlcomm_cnt, submit_cnt, thrid, nthr)

/* available comm slots, one array per thread:
   MUST BE SHARED and long enough so that all threads can keep their
   message status there. Note: other threads can complete the request.
 */
int **available = NULL;

int total_sent = 0;
int ongoing_comm = 0;
int inflight;

void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    available[pthr][pos] = 1;
    comm_cnt++;
}

void recv_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << " pthr " <<  pthr << " ongoing " << ongoing_comm << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    available[pthr][pos] = 1;

#pragma omp atomic
    ongoing_comm--;
    comm_cnt++;
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
	/* TODO this needs to be made per-thread. 
	   If we make 'static' variables, then we can't initialize m_rank and anything else
	   that used MPI in the constructor, as it will be executed before MPI_Init.
	*/
	gridtools::ghex::tl::callback_communicator<CommType> *comm 
	    = new gridtools::ghex::tl::callback_communicator<CommType>();

#pragma omp master
	{
	    rank = comm->m_rank;
	    size = comm->m_size;
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(*comm).name() << "\n\n";
	}

	std::vector<MsgType> msgs;
	
#pragma omp barrier
	comm->whoami();

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size);
	}
	
#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	/** initialize request availability arrays */
#pragma omp master
	available = new int *[nthr];

#pragma omp barrier
#pragma omp flush(available)
	available[thrid] = new int[inflight];
	for(int j=0; j<inflight; j++){
	    available[thrid][j] = 1;
	}

	/* make sure both ranks are started and all threads initialized */
	comm->barrier();

	comm_cnt = 0;
	nlcomm_cnt = 0;
	submit_cnt = 0;

	/* to tackle inacurate message counting, quit the benchmark if we don't see progress */
#define NOPROGRESS_CNT 10000
	int noprogress = 0;
	
	int i = 0, dbg = 0, blk;
	blk = niter / 10;
	dbg = dbg + blk;

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    while(total_sent < niter && noprogress < NOPROGRESS_CNT){
		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){

			// reset the no-progress counter
			noprogress = 0;

			// progress output
			if(rank==0 && thrid==0 && total_sent >= dbg) {
			    std::cout << total_sent << " iters\n";
			    dbg = dbg + blk;
			}

			// don't really need atomic, because we dont' really care about precise number of messages
			// as long as there are more than niter - we need to receive them
			// However, due to counting errors we need to get out of the loop based on a noprogress condition,
			// which would fire when we make counting mistakes.
			// #pragma omp atomic
			total_sent++;

			// number of requests per thread
			submit_cnt++;

			available[thrid][j] = 0;
			comm->send(msgs[j], peer_rank, thrid*inflight+j, send_callback);
		    }
		}
		comm->progress();
		noprogress++;
	    }
	} else {

	    /* recv requests are resubmitted as soon as a request is completed */
	    /* so the number of submitted recv requests is always ~constant (inflight) */
	    /* expect niter messages (i.e., niter recv callbacks) on receiver  */
	    ongoing_comm = niter;
#pragma omp barrier

	    while(ongoing_comm>0 && noprogress < NOPROGRESS_CNT){

		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){

			// reset the no-progress counter
			noprogress = 0;

			// number of requests per thread
			submit_cnt++;

			available[thrid][j] = 0;
			comm->recv(msgs[j], peer_rank, thrid*inflight+j, recv_callback);
		    }
		}

		comm->progress();
		noprogress++;
	    }	    

#pragma omp barrier
#pragma omp master
	    {
		if(ongoing_comm<0) ongoing_comm = 0;
		std::cout << "unreceived messages: " << ongoing_comm << " noprogress " << noprogress << "\n";
	    }
	}

	if(noprogress >= NOPROGRESS_CNT) std::cout << "rank " << rank << " finished: no progress threashold\n";

#pragma omp barrier
	comm->flush();
	comm->barrier();

#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " submitted " << submit_cnt
		  << " serviced " << comm_cnt << ", non-local " << nlcomm_cnt << " completion events\n";
	
	delete comm;
    }
    
    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    // MPI_Finalize();
#endif
}
