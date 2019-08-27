#include <iostream>
#include <vector>

#include "pmi.h"
#include "communicator.hpp"

int main(int argc, char *argvp[])
{
    gridtools::ghex::ucx::communicator::rank_type rank, src, dst, size;
    gridtools::ghex::ucx::communicator::future_type sreq, rreq;
	
    std::vector<unsigned char> smsg = {1,2,3,4,5,6,7,8,9,10};
    std::vector<unsigned char> rmsg(10);

    pmi_init();
    rank = pmi_get_rank();
    size = pmi_get_size();
    
    gridtools::ghex::ucx::communicator comm;

    /* recv from previous, send to next */
    dst = (rank+1)%size;
    src = (rank-1+size)%size;
    std::cout << rank << " send to " << dst << " and recv from " << src << "\n";

    std::cout << rank << " async send\n";
    sreq = comm.send(smsg, dst, 42);
    rreq = comm.recv(rmsg, src, 42);
    rreq.wait();

    std::cout << rank << " blockig send\n";
    comm.blocking_send(smsg, dst, 42);
    rreq = comm.recv(rmsg, src, 42);
    rreq.wait();
    
    pmi_finalize();
}
