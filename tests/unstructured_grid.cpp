
#include "../include/prototype/structured_grid.hpp"
#include "../include/prototype/unstructured_grid.hpp"
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <list>


//using namespace gridtools;
namespace gt = gridtools;

using id_type = int;


namespace gridtools {


template<typename Grid>
struct visit
{
    template<typename F>
    void apply(Grid& g, F&& f)
    {
    }
};

template<typename T>
struct visit<local_unstructured_grid_data<T>>
{
    template<typename F>
    void apply(local_unstructured_grid_data<T>& g, F&& f)
    {
        for (typename local_unstructured_grid_data<T>::cell_id_type id=g.m_begin; id<g.m_end; ++id)
            f(id, get_domain_id(id));

        for (typename local_unstructured_grid_data<T>::cell_id_type id : g.m_recv_index)
            f(id, get_domain_id(id));
    }
};




}




int main(int argc, char** argv)
{
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);

    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::stringstream ss;
    ss << rank;
    std::string filename = "out" + ss.str() + ".txt";
    std::cout << filename << std::endl;
    std::ofstream file(filename.c_str());

    /*
    Here we take a 2D unstructured grid and use a user-defined domain decomposition.
    Each MPI rank has 2 subdomains (6 ranks in total).
    */

    // local ids
    std::list<id_type> local_ids{ rank*2, rank*2+1 };

    file << "Local ids\n";
    std::for_each(local_ids.begin(), local_ids.end(), [&file] (id_type const& x) { file << x << ", ";});
    file << "\n";

    /*// neighbor generator
    auto neighbor_generator = [](id_type id, int r) -> std::vector<std::pair<id_type, int>>
    {
        
    }*/


    gt::local_unstructured_grid_data<int> grid_1(5,rank*2);
    gt::local_unstructured_grid_data<int> grid_2(5,rank*2+1);

    gt::local_structured_grid_data<int> sgrid_1(5,rank*2);
    gt::local_structured_grid_data<int> sgrid_2(5,rank*2+1);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==1)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==2)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    if (rank==0)
    {
        // visit edge neighbors (principal neighbors)
        grid_2.visit_neighbors<0>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
        std::cout << std::endl;
        // visit vertex neighbors
        grid_2.visit_neighbors<1>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
        std::cout.flush();
    }




    MPI_Finalize();
    return 0;
}

