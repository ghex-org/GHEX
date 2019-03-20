
#include "../include/prototype/unstructured_grid.hpp"
#include <mpi.h>
#include <iostream>

using namespace gridtools;

int main(int argc, char** argv)
{
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    local_unstructured_grid_data<int> grid(5,rank);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << grid;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==1)
    {
        std::cout << grid;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==2)
    {
        std::cout << grid;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==3)
    {
        std::cout << grid;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==4)
    {
        std::cout << grid;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);






    MPI_Finalize();
    return 0;
}

