#include <iostream>
#include <vector>
#include "tictoc.h"

#ifdef USE_MPI
#include "communicator_mpi.hpp"
using CommType = gridtools::ghex::mpi::communicator;
using namespace gridtools::ghex::mpi;
#define USE_CALLBACK_COMM
#else
#ifdef USE_UCX_NBR

/* use the GHEX callback framework */
#include "communicator_ucx_nbr.hpp"
#define USE_CALLBACK_COMM
#else

/* use the UCX's own callback framework */
#include "communicator_ucx.hpp"
#undef  USE_CALLBACK_COMM
#endif
using CommType = gridtools::ghex::ucx::communicator;
using namespace gridtools::ghex::ucx;
#endif

#ifdef USE_CALLBACK_COMM
#include "callback_communicator.hpp"
CommType comm;
gridtools::ghex::callback_communicator<CommType> comm_cb(comm);
#else
CommType comm;
#define comm_cb comm
#endif

#include "message.hpp"
using MsgType = gridtools::ghex::mpi::raw_shared_message<>;

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

    niter = atoi(argv[1]);
    buff_size = atoi(argv[2]);
    inflight = atoi(argv[3]);   
    
    rank = comm.m_rank;
    size = comm.m_size;
    peer_rank = (rank+1)%2;

    if(rank==0)	std::cout << "\n\nrunning test " << __FILE__ << " with communicator " << comm.name << "\n\n";

    {
	std::vector<MsgType> msgs;

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size, buff_size);
	}
	
	if(rank == 1) {
	    tic();
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
		    comm_cb.send(msgs[j], 1, j, send_callback);
		else
		    comm_cb.recv(msgs[j], 0, j, recv_callback);
		if(i==niter) break;
	    }
		
	    /* complete all inflight requests before moving on */
	    while(ongoing_comm){
		comm_cb.progress();
	    }
	}

	if(rank == 1) toc();
	comm.fence();
    }
}
