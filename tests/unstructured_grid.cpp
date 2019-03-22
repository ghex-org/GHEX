


#include "../include/prototype/grid.hpp"
#include "../include/prototype/structured_grid.hpp"
#include "../include/prototype/unstructured_grid.hpp"
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <list>
#include <tuple>

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

    gt::local_structured_grid_data<int> sgrid_1(1,rank*2);
    gt::local_structured_grid_data<int> sgrid_2(1,rank*2+1);



    // map function for regular grid map 1
    auto regular_map_function_1 = 
        [&sgrid_1]
        (typename decltype(sgrid_1)::local_cell_id_t loc_id) 
            -> std::tuple<typename decltype(sgrid_1)::global_cell_id_t, typename decltype(sgrid_1)::domain_id_t>
        {
            const auto glob_id = sgrid_1.global_cell_id(loc_id);
            const auto dom_id  = gt::get_structured_domain_id(glob_id);
            return {glob_id, dom_id};
        };

    // regular grid map 1 
    gt::regular_grid_map<
        typename decltype(sgrid_1)::local_cell_id_t,
        typename decltype(sgrid_1)::global_cell_id_t,
        typename decltype(sgrid_1)::domain_id_t,
        typename decltype(sgrid_1)::extent_t,
        decltype(regular_map_function_1)
    > s_map_1(
        1,
        1,
        sgrid_1.m_begin,
        sgrid_1.m_end,
        regular_map_function_1
    );


    // map function for regular grid map 2
    auto regular_map_function_2 = 
        [&sgrid_2]
        (typename decltype(sgrid_2)::local_cell_id_t loc_id) 
            -> std::tuple<typename decltype(sgrid_2)::global_cell_id_t, typename decltype(sgrid_2)::domain_id_t>
        {
            const auto glob_id = sgrid_2.global_cell_id(loc_id);
            const auto dom_id  = gt::get_structured_domain_id(glob_id);
            return {glob_id, dom_id};
        };

    // regular grid map 2 
    gt::regular_grid_map<
        typename decltype(sgrid_2)::local_cell_id_t,
        typename decltype(sgrid_2)::global_cell_id_t,
        typename decltype(sgrid_2)::domain_id_t,
        typename decltype(sgrid_2)::extent_t,
        decltype(regular_map_function_2)
    > s_map_2(
        1,
        1,
        sgrid_2.m_begin,
        sgrid_2.m_end,
        regular_map_function_2
    );
    

    
    /*MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        for (const auto& x : s_map_1.m_recv_ranges)
        {
            std::cout << "neighbor recv domain " << x.first << std::endl;
            for (const auto& r : x.second)
            {
                std::cout << "  (" 
                << r.m_begin[0] << ", "
                << r.m_begin[1] << "), ("
                << r.m_end[0] << ", "
                << r.m_end[1] << "), "
                << r.m_first << ", "
                << r.m_last
                << std::endl;
            }
        }
        std::cout << std::endl;
        for (const auto& x : s_map_1.m_send_ranges)
        {
            std::cout << "neighbor send domain " << x.first << std::endl;
            for (const auto& r : x.second)
            {
                std::cout << "  (" 
                << r.m_begin[0] << ", "
                << r.m_begin[1] << "), ("
                << r.m_end[0] << ", "
                << r.m_end[1] << "), "
                << r.m_first << ", "
                << r.m_last
                << std::endl;
            }
        }
        std::cout.flush();
    }*/


    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        //std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        //std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==1)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        //std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        //std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==2)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        //std::cout << grid_1;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        //std::cout << grid_2;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /*if (rank==0)
    {
        // visit edge neighbors (principal neighbors)
        grid_2.visit_neighbors<0>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
        std::cout << std::endl;
        // visit vertex neighbors
        grid_2.visit_neighbors<1>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
        std::cout.flush();
    }*/


    // pack send values
    std::vector<std::vector<int>> s_send_package_1;
    s_map_1.pack(s_send_package_1, sgrid_1);

    std::vector<std::vector<int>> s_send_package_2;
    s_map_2.pack(s_send_package_2, sgrid_2);

    /*MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << "send package 1" << std::endl;
        for (const auto& v : s_send_package_1)
        {
            for (const auto& x : v)
                std::cout << x << " ";
            std::cout << std::endl;
        }
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);*/

    auto s_domain_id_to_rank = [](int domain_id) { return domain_id/2; };

    std::vector<std::vector<int>> s_recv_package_1;
    auto reqs_1 = s_map_1.exchange(s_send_package_1, s_recv_package_1, s_domain_id_to_rank);

    std::vector<std::vector<int>> s_recv_package_2;
    auto reqs_2 = s_map_2.exchange(s_send_package_2, s_recv_package_2, s_domain_id_to_rank);

    

    for (auto& req : reqs_1)
    {
            MPI_Status st;
            MPI_Wait(&req, &st);
    }
    for (auto& req : reqs_2)
    {
            MPI_Status st;
            MPI_Wait(&req, &st);
    }

    /*MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << "recv package 1" << std::endl;
        for (const auto& v : s_recv_package_1)
        {
            for (const auto& x : v)
                std::cout << x << " ";
            std::cout << std::endl;
        }
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);*/
    

    // unpack
    s_map_1.unpack(s_recv_package_1, sgrid_1);
    s_map_2.unpack(s_recv_package_2, sgrid_2);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==1)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==2)
    {
        std::cout << "rank = " << rank << std::endl;
        std::cout << "  id = " << rank*2 << std::endl;
        std::cout << sgrid_1;
        std::cout << "  id = " << rank*2+1 << std::endl;
        std::cout << sgrid_2;
        std::cout.flush();
    }
    MPI_Barrier(MPI_COMM_WORLD);
   

    MPI_Finalize();
    return 0;
}

