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
int comm_cnt = 0, nlcomm_cnt = 0, submit_cnt = 0;
int thrid, nthr;
#pragma omp threadprivate(comm_cnt, nlcomm_cnt, submit_cnt, thrid, nthr)

int ongoing_comm = 0;
int inflight;

void send_callback(MsgType mesg, int rank, int tag)
{
    // std::cout << "send callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlcomm_cnt++;

#pragma omp atomic
    ongoing_comm--;
    comm_cnt++;
}

void recv_callback(MsgType mesg, int rank, int tag)
{
    //std::cout << "recv callback called " << rank << " thread " << omp_get_thread_num() << " tag " << tag << "\n";
    int pthr = tag/inflight;
    if(pthr != thrid) nlcomm_cnt++;

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

	for(int j=0; j<inflight; j++){
	    msgs.emplace_back(buff_size);
	}
	
#pragma omp barrier
#pragma omp master
	if(rank == 1) {
	    timer.tic();
	    bytes = (double)niter*size*buff_size/2;
	}
	
	thrid = omp_get_thread_num();
	nthr = omp_get_num_threads();

	/* make sure both ranks are started and all threads initialized */
	comm->barrier();

	int i = 0, dbg = 0;

	/* send / recv niter messages, work in inflight requests at a time */
	while(i<niter){

#pragma omp atomic
	    ongoing_comm += inflight;

	    /* submit inflight requests */
	    for(int j=0; j<inflight; j++){
		if(rank==0 && thrid==0 && dbg>=(niter/10)) {
		    std::cout << i << " iters\n";
		    dbg=0;
		}
		submit_cnt++;
		i += nthr;
		dbg += nthr; 
		if(rank==0)
		    comm->send(msgs[j], peer_rank, thrid*inflight+j, send_callback);
		else
		    comm->recv(msgs[j], peer_rank, thrid*inflight+j, recv_callback);
		if(i >= niter) break;
	    }

	    /* complete all inflight requests before moving on */
	    while(ongoing_comm>0){
		comm->progress();
	    }
	}

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
    MPI_Finalize();
#endif
}
