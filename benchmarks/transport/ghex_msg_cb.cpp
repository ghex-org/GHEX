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


/* comm requests currently in-flight */
int ongoing_comm = 0;

void send_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    ongoing_comm--;
}

void recv_callback(int rank, int tag, const MsgType &mesg)
{
    // std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
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
    
    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << typeid(comm).name() << "\n\n";

    {
	std::vector<MsgType> msgs;
	gridtools::ghex::timer timer;
	long bytes = 0;
	
	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size);
	}
	
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}

	/* send / recv niter messages, work in inflight requests at a time */
	int i = 0;
	while(i<niter){

	    /* submit inflight requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && (i)%(niter/10)==0) fprintf(stderr, "%d iters\n", i);
		i++;
		ongoing_comm++;
		if(rank==0)
		    comm_cb.send(1, j, msgs[j], send_callback);
		else
		    comm_cb.recv(0, j, msgs[j], recv_callback);
		if(i==niter) break;
	    }
		
	    /* complete all inflight requests before moving on */
	    while(ongoing_comm){
		comm_cb.progress();
	    }
	}

	if(rank == 1) timer.vtoc(bytes);
	// comm.fence();
    }

#ifdef USE_MPI
    MPI_Finalize();
#endif
}
