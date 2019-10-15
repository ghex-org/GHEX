#include <iostream>
#include <vector>

/* define to use the raw shared message - lower overhead */
#define GHEX_USE_RAW_SHARED_MESSAGE

#include <ghex/common/timer.hpp>
#include <ghex/transport_layer/callback_communicator.hpp>
using MsgType = gridtools::ghex::tl::shared_message_buffer<>;


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

/* available comm slots */
int *available = NULL;
int ongoing_comm = 0;

void send_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    available[tag] = 1;
    ongoing_comm--;
}

#ifdef USE_CALLBACK_COMM
gridtools::ghex::tl::callback_communicator<CommType> *p_comm_cb = NULL;
#else
CommType *p_comm_cb = NULL;
#endif

void recv_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    p_comm_cb->recv(rank, tag, (MsgType&)mesg, recv_callback);
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
    gridtools::ghex::tl::callback_communicator<CommType> comm_cb(comm);
#else
#define comm_cb comm
#endif
    p_comm_cb = &comm_cb;

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";

    {
	gridtools::ghex::timer timer;
	long bytes = 0;
	std::vector<MsgType> msgs;
	available = new int[inflight];

	for(int j=0; j<inflight; j++){
	    available[j] = 1;
	    msgs.emplace_back(buff_size);
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
			if(rank==0 && (sent)%(niter/10)==0) fprintf(stderr, "%d iters\n", sent);
			available[j] = 0;
			sent++;
			ongoing_comm++;
			comm_cb.send(1, j, msgs[j], send_callback);
			if(sent==niter) break;
		    }
		}
		if(sent==niter) break;
	    
		/* progress a bit: for large inflight values this yields better performance */
		/* over simply calling the progress once */
		/* TODO: optimization target: funny that the below loop is faster for 1k inflight */
		// for(int i=0; i<100; i++) comm.progress();
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

	    /* submit all recv requests */
	    for(int j=0; j<inflight; j++){
		comm_cb.recv(0, j, msgs[j], recv_callback);
	    }

	    /* requests are re-submitted inside the calback. */
	    /* progress (below) until niter messages have been received. */
	}

	/* complete all comm */
	while(ongoing_comm){
	    comm_cb.progress();
	}

	if(rank == 1) timer.vtoc(bytes);
	// comm.fence();
    }

#ifdef USE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
#endif
}
