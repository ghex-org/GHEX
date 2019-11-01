#include <iostream>
#include <vector>

#include <ghex/common/timer.hpp>

#ifdef USE_POOL_ALLOCATOR
#include "pool_allocator.hpp"
using AllocType = ghex::allocator::pool_allocator<unsigned char, std::allocator<unsigned char>>;
#else
using AllocType = std::allocator<unsigned char>;
#endif

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

using MsgType = gridtools::ghex::tl::shared_message_buffer<AllocType>;

/* Track finished comm requests. 
   This is shared between threads, because in the shared-worker case
   there is no way of knowing which thread will service which requests,
   and how many.
*/
int comm_cnt = 0, nlcomm_cnt = 0, submit_cnt = 0;
int thrid, nthr;
#pragma omp threadprivate(comm_cnt, nlcomm_cnt, submit_cnt, thrid, nthr)

/* available comm slots - per-thread */
int **available = NULL;
int ongoing_comm = 0;
int inflight;

void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    comm_cnt++;
    available[pthr][pos] = 1;
}

gridtools::ghex::tl::callback_communicator<CommType, AllocType> *pcomm = NULL;
#pragma omp threadprivate(pcomm)

void recv_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << " ongoing " << ongoing_comm << "\n";
    int pthr = tag/inflight;
    int pos = tag - pthr*inflight;
    if(pthr != thrid) nlcomm_cnt++;
    comm_cnt++;
    submit_cnt+=nthr;

    /* resubmit the recv request */
    pcomm->recv(mesg, rank, tag, recv_callback);

#pragma omp atomic
    ongoing_comm--;
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
	gridtools::ghex::tl::callback_communicator<CommType, AllocType> *comm 
	    = new gridtools::ghex::tl::callback_communicator<CommType, AllocType>();
	AllocType alloc;

#pragma omp master
	{
	    rank = comm->rank();
	    size = comm->size();
	    peer_rank = (rank+1)%2;
	    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(*comm).name() << "\n\n";
	}

	/* needed in the recv_callback to resubmit the recv request */
	pcomm = comm;

	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

#pragma omp master
	available = new int*[nthr];
#pragma omp barrier
	available[thrid] = new int[inflight];

	for(int j=0; j<inflight; j++){
	    available[thrid][j] = 1;
	}

	/* make sure both ranks are started and all threads initialized */
	comm->barrier();
	
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	int i = 0, dbg = 0, blk;
	blk = niter / 10;
	dbg = dbg + blk;

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    while(submit_cnt < niter){
		
		for(int j=0; j<inflight; j++){
		    if(available[thrid][j]){
			if(rank==0 && thrid==0 && submit_cnt >= dbg) {
			    std::cout << submit_cnt << " iters\n";
			    dbg = dbg + blk;
			}
			available[thrid][j] = 0;
			submit_cnt += nthr;
			MsgType msg = MsgType(buff_size, alloc);
			comm->send(msg, peer_rank, thrid*inflight+j, send_callback);
		    }
		}
		comm->progress();
	    }

	} else {

	    /* recv requests are resubmitted as soon as a request is completed */
	    /* so the number of submitted recv requests is always constant (inflight) */
	    /* expect niter messages (i.e., niter recv callbacks) on receiver  */
	    ongoing_comm = niter;
#pragma omp barrier

	    /* submit all recv requests */
	    for(int j=0; j<inflight; j++){
		MsgType msg = MsgType(buff_size, alloc);
		comm->recv(msg, peer_rank, thrid*inflight+j, recv_callback);
		submit_cnt+=nthr;
	    }
	    
	    /* requests are re-submitted inside the calback. */
	    /* progress (below) until niter messages have been received. */

	    /* complete all comm */
	    while(ongoing_comm>0){
		comm->progress();
	    }
	}

#pragma omp barrier
	comm->flush();
	comm->barrier();
	
#pragma omp critical
	std::cout << "rank " << rank << " thread " << thrid << " submitted " << submit_cnt/nthr
		  << " serviced " << comm_cnt << ", non-local " << nlcomm_cnt << " completion events\n";
	
	delete comm;
    }

    if(rank == 1) timer.vtoc(bytes);

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}
