#include <iostream>
#include <vector>
#include <unistd.h>

#ifdef USE_POOL_ALLOCATOR
#include "pool_allocator.hpp"
using AllocType = ghex::allocator::pool_allocator<unsigned char, std::allocator<unsigned char>>;
#else
using AllocType = std::allocator<unsigned char>;
#endif

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/callback_communicator.hpp>
using MsgType = gridtools::ghex::tl::shared_message_buffer<AllocType>;


#ifdef USE_MPI

/* MPI backend */
#include <ghex/transport_layer/mpi/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag>;
#define USE_CALLBACK_COMM
#else

/* UCX backend */
#include <ghex/transport_layer/ucx/communicator.hpp>
using CommType = gridtools::ghex::tl::communicator<gridtools::ghex::tl::ucx_tag>;

#ifdef USE_UCX_NBR
/* use the GHEX callback framework */
#define USE_CALLBACK_COMM
#else
/* use the UCX's own callback framework */
#include <ghex/transport_layer/ucx/communicator.hpp>
#undef  USE_CALLBACK_COMM
#endif /* USE_UCX_NBR */

#endif /* USE_MPI */

AllocType alloc;
int grank;

/* available comm slots */
int *available = NULL;
int ongoing_comm = 0;

void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;
}

void recv_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;
}

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

#ifdef USE_CALLBACK_COMM
    gridtools::ghex::tl::callback_communicator<CommType, AllocType> comm_cb(comm);
#else
#define comm_cb comm
#endif

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    grank = rank;

#ifdef USE_POOL_ALLOCATOR
    alloc.initialize(inflight+1, buff_size);
#endif
    
    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";

    {
	gridtools::ghex::timer timer;
	long bytes = 0;
	available = new int[inflight];

	for(int j=0; j<inflight; j++){
	    available[j] = 1;
	}

	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	if(rank == 0){

	    /* send niter messages - as soon as a slot becomes free */
	    int sent = 0;
	    while(sent != niter){
		
		for(int j=0; j<inflight; j++){
		    if(available[j]){
			if(rank==0 && (sent)%(niter/10)==0) {
			    std::cout << sent << " iters\n";
			}
			available[j] = 0;
			sent++;
			ongoing_comm++;
			MsgType msg = MsgType(buff_size, alloc);
			comm_cb.send(msg, peer_rank, j, send_callback);
			if(sent==niter) break;
		    }
		}
		if(sent==niter) break;
	    
		/* progress a bit: for large inflight values this yields better performance */
		/* over simply calling the progress once */
		int p = 0.1*inflight-1;
		do {
		    p-=comm_cb.progress();
		} while(ongoing_comm && p>0);
	    }

	} else {

	    /* recv requests are resubmitted as soon as a request is completed */
	    /* so the number of submitted recv requests is always constant (inflight) */
	    /* expect niter messages (i.e., niter recv callbacks) on receiver  */
	    ongoing_comm = niter;
	    while(ongoing_comm){
		
		for(int j=0; j<inflight; j++){
		    if(available[j]){
			available[j] = 0;
			MsgType msg = MsgType(buff_size, alloc);
			comm_cb.recv(msg, peer_rank, j, recv_callback);
		    }
		}
	    
		/* progress a bit: for large inflight values this yields better performance */
		/* over simply calling the progress once */
		int p = 0.1*inflight-1;
		do {
		    p-=comm_cb.progress();
		} while(ongoing_comm && p>0);
	    }	    
	}

	/* complete all comm */
	while(ongoing_comm){
	    comm_cb.progress();
	}

	comm.fence();
	comm.barrier();
	
	if(rank == 1) timer.vtoc(bytes);
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}
