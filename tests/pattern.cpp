#include "../include/pattern.hpp"
#include <boost/mpi/environment.hpp>
#include <iostream>
#include <array>

struct my_domain_desc
{
    using coordinate_type = std::array<int,3>;
    using domain_id_type  = int;
};

bool test0(boost::mpi::communicator& mpi_comm)
{
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};

    std::vector<my_domain_desc> local_domains(4);

    auto patterns = gridtools::make_pattern<gridtools::protocol::mpi, gridtools::structured_grid>(comm, local_domains);

    return true;
}

int main(int argc, char* argv[])
{
    //MPI_Init(&argc,&argv);
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;


    auto passed = test0(world);

    return 0;
}
