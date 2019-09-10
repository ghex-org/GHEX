#include <iostream>
#include <vector>

#include "pmi.h"
#include "communicator.hpp"

int ready_comm = 0;
void send_callback(int rank, int tag, void *mesg)
{
    std::cout << "send callback called " << rank << " tag " << tag << "\n";
    ready_comm++;
}

void recv_callback(int rank, int tag, void *mesg)
{
    std::cout << "recv callback called " << rank << " tag " << tag << "\n";
    ready_comm++;
}

int main(int argc, char *argvp[])
{
    gridtools::ghex::ucx::communicator::rank_type rank, src, dst, size;
    gridtools::ghex::ucx::communicator::future_type sreq, rreq;
	
    std::vector<unsigned char> smsg(1024*1024);
    std::vector<unsigned char> rmsg(1024*1024);

    pmi_init();
    rank = pmi_get_rank();
    size = pmi_get_size();
    
    gridtools::ghex::ucx::communicator comm;

    /* recv from previous, send to next */
    dst = (rank+1)%size;
    src = (rank-1+size)%size;
    std::cout << rank << " send to " << dst << " and recv from " << src << "\n";

    /* callback based comm */
    std::cout << rank << " send / recv with callback\n";
    ready_comm = 0;
    comm.send(smsg, dst, 42, send_callback);
    comm.recv(rmsg, src, 42, recv_callback);

    while(ready_comm != 2){
    	comm.progress();
    }

    /* async comm */
    std::cout << rank << " async send with future\n";
    sreq = comm.send(smsg, dst, 42);
    rreq = comm.recv(rmsg, src, 42);
    rreq.wait();

    /* blocking send */
    /* carefull: must be after a recv has been preposted! */
    std::cout << rank << " blockig send\n";
    rreq = comm.recv(rmsg, src, 42);
    comm.blocking_send(smsg, dst, 42);
    rreq.wait();
    
    pmi_finalize();
}
