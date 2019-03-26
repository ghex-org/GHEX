


#include "../include/prototype/grid.hpp"
#include "../include/prototype/structured_grid.hpp"
#include "../include/prototype/unstructured_grid.hpp"
#include <mpi.h>
#include <iostream>
#include <sstream>
#include <fstream>
#include <list>
#include <tuple>
#include <thread>
#include <mutex>


//using namespace gridtools;
namespace gt = gridtools;

template<typename Id>
struct unstructured_range
{
    struct iterator
    {
        Id m_begin;
        Id m_end;
        std::vector<Id> const * m_index_vec_ptr;
        Id m_inner_pos;
        int m_outer_pos;

        iterator(Id begin, Id end, std::vector<Id> const * ptr, Id inner_pos, int outer_pos = -1) noexcept :
            m_begin(begin),
            m_end(end),
            m_index_vec_ptr(ptr),
            m_inner_pos(inner_pos),
            m_outer_pos(outer_pos)
        {
            if (outer_pos>=0)
                m_inner_pos = m_end;
        }

        iterator(const iterator&) noexcept = default;
        iterator(iterator&&) noexcept = default;
        iterator& operator=(const iterator&) noexcept = default;
        iterator& operator=(iterator&&) noexcept = default;

        void swap(iterator& other) noexcept
        {
            std::swap(m_begin, other.m_begin);
            std::swap(m_end, other.m_end);
            std::swap(m_index_vec_ptr, other.m_index_vec_ptr);
            std::swap(m_inner_pos, other.m_inner_pos);
            std::swap(m_outer_pos, other.m_outer_pos);
        }

        iterator& operator++() noexcept
        {
            if (inner())
            {
                ++m_inner_pos;
                if (m_inner_pos == m_end)
                    m_outer_pos = 0;
            }
            else
            {
                if (m_outer_pos < (int)(m_index_vec_ptr->size()))
                    ++m_outer_pos;
            }
            return *this;
        }

        iterator operator++(int) const noexcept
        {
            iterator tmp(*this);
            operator++();
            return tmp;
        }

        Id operator*() const noexcept
        {
            return inner() ? m_inner_pos : m_index_vec_ptr->operator[](m_outer_pos);
        }

        bool operator!=(const iterator& other) const noexcept
        {
            return !operator==(other);
        }

        bool operator==(const iterator& other) const noexcept
        {
            return     (m_begin         == other.m_begin)
                    && (m_end           == other.m_end)
                    && (m_index_vec_ptr == other.m_index_vec_ptr)
                    && (m_inner_pos     == other.m_inner_pos)
                    && (m_outer_pos     == other.m_outer_pos);
        }

        bool operator<(const iterator& other) const noexcept
        {
            if (inner())
            {
                if (other.inner())
                    return m_inner_pos < m_outer_pos;
                else
                    return true;
            }
            else
            {
                if (other.inner())
                    return false;
                else
                    return m_outer_pos < other.m_outer_pos;
            }
        }

        bool inner() const noexcept
        {
            return m_outer_pos < 0;
        }
    };

    iterator m_begin;
    iterator m_end;

    template<typename Grid>
    unstructured_range(const Grid& g) noexcept :
        m_begin(g.m_begin, g.m_end, &g.m_recv_index, g.m_begin),
        m_end(g.m_begin, g.m_end, &g.m_recv_index, g.m_end, g.m_recv_index.size())
    {}

    iterator begin() const noexcept { return m_begin; }
    iterator end() const noexcept { return m_end; }
};

void test_unstructured_grids_serial(int rank)
{
    // 2 domains
    std::vector<int> domain_ids;
    domain_ids.push_back(rank*2);
    domain_ids.push_back(rank*2+1);   
    
    // 2 unsstructured grids    
    using grid_t = gt::local_unstructured_grid_data<int>;
    std::array<grid_t,2> grids{grid_t(1,domain_ids[0]), grid_t(1,domain_ids[1])};

    // print grid data
    for (int r = 0; r<3; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r==rank)
        {
            std::cout << "rank = " << rank << std::endl;
            std::cout << "--------" << std::endl;
            auto it_d = domain_ids.begin();
            for (auto it = grids.begin(); it!=grids.end(); ++it_d, ++it)
            {
                std::cout << "  id = " << *it_d << std::endl;
                std::cout << *it;
                std::cout << std::endl;
            }
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // unstructured grid map 
    using grid_map_t = gt::unstructured_grid_map<
        typename grid_t::cell_id_type,
        typename grid_t::cell_id_type,
        int,
        int>;
    std::vector<grid_map_t> maps;

    using range_t = unstructured_range<typename grid_t::cell_id_type>;

    /*MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        range_t r(grids[0]);
        for (auto idx : r)
            std::cout << idx << std::endl;
        std::cout << std::endl;
        std::cout.flush();
    }*/

    // create maps
    auto it_d = domain_ids.begin();
    for (auto it=grids.begin(); it!=grids.end(); ++it_d, ++it)
    {
        const auto& grid(*it);
        maps.emplace_back(
            grid_map_t(
                1,                                        // vertical size
                *it_d,                                    // domain id
                range_t(grid),                            // local index range
                [&grid](typename grid_t::cell_id_type id) // neighbor index range generator
                { 
                    return grid.get_neighbor_indices(id); 
                },
                [](typename grid_t::cell_id_type id)      // global map: local_id -> {global_id, domain_id}
                { 
                    return std::make_tuple(id, gt::get_domain_id(id)); 
                } 
            )
        );
    }

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        for (const auto& p: maps[0].m_send_ids)
        {
            std::cout << "send to " << p.first << ":" << std::endl;
            for (const auto& i : p.second)
                std::cout << i << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        for (const auto& p: maps[0].m_recv_ids)
        {
            std::cout << "recv from " << p.first << ":" << std::endl;
            for (const auto& i : p.second)
                std::cout << i << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout.flush();
    }

    // prepare temporary storage
    std::array<std::vector<std::vector<int>>,2> send_package;
    std::array<std::vector<std::vector<int>>,2> recv_package;
    std::vector<MPI_Request> reqs;

    // pack data
    auto it_s = send_package.begin();
    auto it_g = grids.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_g, ++it_s, ++it)
        it->pack(*it_s, *it_g);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "send package: " << std::endl;
        for (const auto& v: send_package[0])
        {
            for (const auto& x : v)
                std::cout << x << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout.flush();
    }

    // exchange data
    it_s = send_package.begin();
    auto it_r = recv_package.begin();
    auto it_req = reqs.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_req, ++it_r, ++it_s, ++it)
    {
        auto reqs_i = it->exchange(*it_s, *it_r, [](int domain_id)->int { return domain_id/2;});
        reqs.insert(reqs.end(), reqs_i.begin(), reqs_i.end());
    }

    // wait for exchange to finish
    std::vector<MPI_Status> sts(reqs.size());
    MPI_Waitall(reqs.size(), &reqs[0], &sts[0]);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0)
    {
        std::cout << "recv package: " << std::endl;
        for (const auto& v: recv_package[0])
        {
            for (const auto& x : v)
                std::cout << x << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;
        std::cout.flush();
    }

    // unpack data
    it_r = recv_package.begin();
    it_g = grids.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_g, ++it_r, ++it)
        it->unpack(*it_r,*it_g);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << std::endl << "=========================" << std::endl << std::endl;
        std::cout.flush();
    }

    // print grid data
    for (int r = 0; r<3; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r==rank)
        {
            std::cout << "rank = " << rank << std::endl;
            std::cout << "--------" << std::endl;
            auto it_d = domain_ids.begin();
            for (auto it = grids.begin(); it!=grids.end(); ++it_d, ++it)
            {
                std::cout << "  id = " << *it_d << std::endl;
                std::cout << *it;
                std::cout << std::endl;
            }
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
}

void test_regular_grids_serial(int rank)
{
    // 2 domains
    std::vector<int> domain_ids;
    domain_ids.push_back(rank*2);
    domain_ids.push_back(rank*2+1);   
    
    // 2 regular grids    
    using grid_t = gt::local_structured_grid_data<int>;
    std::array<grid_t,2> grids{grid_t(1,domain_ids[0]), grid_t(1,domain_ids[1])};

    // print grid data
    for (int r = 0; r<3; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r==rank)
        {
            std::cout << "rank = " << rank << std::endl;
            std::cout << "--------" << std::endl;
            auto it_d = domain_ids.begin();
            for (auto it = grids.begin(); it!=grids.end(); ++it_d, ++it)
            {
                std::cout << "  id = " << *it_d << std::endl;
                std::cout << *it;
                std::cout << std::endl;
            }
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    // regular grid map 
    using grid_map_t = gt::regular_grid_map<
        typename grid_t::local_cell_id_t,
        typename grid_t::global_cell_id_t,
        typename grid_t::domain_id_t,
        typename grid_t::extent_t>;
    std::vector<grid_map_t> maps;

    // create maps
    for (const auto& grid : grids)
    {
        maps.emplace_back(
            grid_map_t(
                1,
                1,
                grid.m_begin, 
                grid.m_end,
                [&grid] (typename grid_t::local_cell_id_t loc_id) -> std::tuple<typename grid_t::global_cell_id_t, typename grid_t::domain_id_t>
                {
                    const auto glob_id = grid.global_cell_id(loc_id);
                    const auto dom_id  = gt::get_structured_domain_id(glob_id);
                    return {glob_id, dom_id};
                }
            )
        );
    }

    // prepare temporary storage
    std::array<std::vector<std::vector<int>>,2> send_package;
    std::array<std::vector<std::vector<int>>,2> recv_package;
    std::vector<MPI_Request> reqs;

    // pack data
    auto it_s = send_package.begin();
    auto it_g = grids.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_g, ++it_s, ++it)
        it->pack(*it_s, *it_g);

    // exchange data
    it_s = send_package.begin();
    auto it_r = recv_package.begin();
    auto it_req = reqs.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_req, ++it_r, ++it_s, ++it)
    {
        auto reqs_i = it->exchange(*it_s, *it_r, [](typename grid_t::domain_id_t domain_id)->int { return domain_id/2;});
        reqs.insert(reqs.end(), reqs_i.begin(), reqs_i.end());
    }

    // wait for exchange to finish
    std::vector<MPI_Status> sts(reqs.size());
    MPI_Waitall(reqs.size(), &reqs[0], &sts[0]);

    // unpack data
    it_r = recv_package.begin();
    it_g = grids.begin();
    for (auto it=maps.begin(); it!=maps.end(); ++it_g, ++it_r, ++it)
        it->unpack(*it_r,*it_g);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank==0)
    {
        std::cout << std::endl << "=========================" << std::endl << std::endl;
        std::cout.flush();
    }

    // print grid data
    for (int r = 0; r<3; ++r)
    {
        MPI_Barrier(MPI_COMM_WORLD);
        if (r==rank)
        {
            std::cout << "rank = " << rank << std::endl;
            std::cout << "--------" << std::endl;
            auto it_d = domain_ids.begin();
            for (auto it = grids.begin(); it!=grids.end(); ++it_d, ++it)
            {
                std::cout << "  id = " << *it_d << std::endl;
                std::cout << *it;
                std::cout << std::endl;
            }
            std::cout.flush();
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

}


void test_regular_grids(int rank)
{
    std::mutex t_mutex;

    // 2 domains
    std::vector<int> domain_ids;
    domain_ids.push_back(rank*2);
    domain_ids.push_back(rank*2+1);   
    
    // 2 regular grids    
    using grid_t = gt::local_structured_grid_data<int>;
    std::array<grid_t,2> grids{grid_t(1,domain_ids[0]), grid_t(1,domain_ids[1])};

    // thread container
    std::vector<std::thread> threads;
    auto itc = domain_ids.begin();
    for (auto it = grids.begin(); it != grids.end(); ++it, ++itc) 
    {    
        threads.push_back(
            std::thread(
                [rank,&t_mutex](grid_t& grid, typename grid_t::domain_id_t dom_id) -> void
                {
                    // map function for regular grid map
                    auto map_function = 
                        [&grid]
                        (typename grid_t::local_cell_id_t loc_id) 
                            -> std::tuple<typename grid_t::global_cell_id_t, typename grid_t::domain_id_t>
                        {
                            const auto glob_id = grid.global_cell_id(loc_id);
                            const auto dom_id  = gt::get_structured_domain_id(glob_id);
                            return {glob_id, dom_id};
                        };

                    // regular grid map 
                    gt::regular_grid_map<
                        typename grid_t::local_cell_id_t,
                        typename grid_t::global_cell_id_t,
                        typename grid_t::domain_id_t,
                        typename grid_t::extent_t
                    > map(1, 1, grid.m_begin, grid.m_end, map_function);

                    // print grid data
                    for (int r = 0; r<3; ++r)
                    {
                        //MPI_Barrier(MPI_COMM_WORLD);
                        std::lock_guard<std::mutex> lock(t_mutex);
                        if (r==rank)
                        {
                            std::cout << "rank = " << rank << std::endl;
                            std::cout << "  id = " << dom_id << std::endl;
                            std::cout << grid;
                            std::cout.flush();
                        }
                        //MPI_Barrier(MPI_COMM_WORLD);
                    }
                    
                    // pack send data
                    std::vector<std::vector<int>> send_package;
                    map.pack(send_package, grid);

                    // domain id map function
                    auto domain_map_function = [](int domain_id) { return domain_id/2; };

                    // exchange data
                    std::vector<std::vector<int>> recv_package;
                    std::vector<MPI_Request> reqs;
                    { 
                        // my MPI library is not thread safe ??
                        std::lock_guard<std::mutex> lock(t_mutex); 
                        reqs = map.exchange(send_package, recv_package, domain_map_function);
                    }

                    // wait for communication to finish
                    for (auto& req : reqs)
                    {
                        MPI_Status st;
                        MPI_Wait(&req, &st);
                    }

                    // unpack data
                    map.unpack(recv_package,grid);

                    // print grid data
                    for (int r = 0; r<3; ++r)
                    {
                        //MPI_Barrier(MPI_COMM_WORLD);
                        std::lock_guard<std::mutex> lock(t_mutex);
                        if (r==rank)
                        {
                            std::cout << "rank = " << rank << std::endl;
                            std::cout << "  id = " << dom_id << std::endl;
                            std::cout << grid;
                            std::cout.flush();
                        }
                        //MPI_Barrier(MPI_COMM_WORLD);
                    }

                },
                std::ref(*it), 
                *itc
            )
        );
    }

    for (auto& th : threads)
        th.join();
}




using id_type = int;


int main(int argc, char** argv)
{
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);
    //MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &p);

    int rank;
    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    //test_regular_grids(rank);
    //test_regular_grids_serial(rank);

    test_unstructured_grids_serial(rank);

//    std::stringstream ss;
//    ss << rank;
//    std::string filename = "out" + ss.str() + ".txt";
//    std::cout << filename << std::endl;
//    std::ofstream file(filename.c_str());
//
//    /*
//    Here we take a 2D unstructured grid and use a user-defined domain decomposition.
//    Each MPI rank has 2 subdomains (6 ranks in total).
//    */
//
//    // local ids
//    std::list<id_type> local_ids{ rank*2, rank*2+1 };
//
//    file << "Local ids\n";
//    std::for_each(local_ids.begin(), local_ids.end(), [&file] (id_type const& x) { file << x << ", ";});
//    file << "\n";
//
//    /*// neighbor generator
//    auto neighbor_generator = [](id_type id, int r) -> std::vector<std::pair<id_type, int>>
//    {
//        
//    }*/
//
//
//    gt::local_unstructured_grid_data<int> grid_1(5,rank*2);
//    gt::local_unstructured_grid_data<int> grid_2(5,rank*2+1);
//
//    gt::local_structured_grid_data<int> sgrid_1(1,rank*2);
//    gt::local_structured_grid_data<int> sgrid_2(1,rank*2+1);
//
//
//
//    // map function for regular grid map 1
//    auto regular_map_function_1 = 
//        [&sgrid_1]
//        (typename decltype(sgrid_1)::local_cell_id_t loc_id) 
//            -> std::tuple<typename decltype(sgrid_1)::global_cell_id_t, typename decltype(sgrid_1)::domain_id_t>
//        {
//            const auto glob_id = sgrid_1.global_cell_id(loc_id);
//            const auto dom_id  = gt::get_structured_domain_id(glob_id);
//            return {glob_id, dom_id};
//        };
//
//    // regular grid map 1 
//    gt::regular_grid_map<
//        typename decltype(sgrid_1)::local_cell_id_t,
//        typename decltype(sgrid_1)::global_cell_id_t,
//        typename decltype(sgrid_1)::domain_id_t,
//        typename decltype(sgrid_1)::extent_t,
//        decltype(regular_map_function_1)
//    > s_map_1(
//        1,
//        1,
//        sgrid_1.m_begin,
//        sgrid_1.m_end,
//        regular_map_function_1
//    );
//
//
//    // map function for regular grid map 2
//    auto regular_map_function_2 = 
//        [&sgrid_2]
//        (typename decltype(sgrid_2)::local_cell_id_t loc_id) 
//            -> std::tuple<typename decltype(sgrid_2)::global_cell_id_t, typename decltype(sgrid_2)::domain_id_t>
//        {
//            const auto glob_id = sgrid_2.global_cell_id(loc_id);
//            const auto dom_id  = gt::get_structured_domain_id(glob_id);
//            return {glob_id, dom_id};
//        };
//
//    // regular grid map 2 
//    gt::regular_grid_map<
//        typename decltype(sgrid_2)::local_cell_id_t,
//        typename decltype(sgrid_2)::global_cell_id_t,
//        typename decltype(sgrid_2)::domain_id_t,
//        typename decltype(sgrid_2)::extent_t,
//        decltype(regular_map_function_2)
//    > s_map_2(
//        1,
//        1,
//        sgrid_2.m_begin,
//        sgrid_2.m_end,
//        regular_map_function_2
//    );
//    
//
//    
//    /*MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0)
//    {
//        for (const auto& x : s_map_1.m_recv_ranges)
//        {
//            std::cout << "neighbor recv domain " << x.first << std::endl;
//            for (const auto& r : x.second)
//            {
//                std::cout << "  (" 
//                << r.m_begin[0] << ", "
//                << r.m_begin[1] << "), ("
//                << r.m_end[0] << ", "
//                << r.m_end[1] << "), "
//                << r.m_first << ", "
//                << r.m_last
//                << std::endl;
//            }
//        }
//        std::cout << std::endl;
//        for (const auto& x : s_map_1.m_send_ranges)
//        {
//            std::cout << "neighbor send domain " << x.first << std::endl;
//            for (const auto& r : x.second)
//            {
//                std::cout << "  (" 
//                << r.m_begin[0] << ", "
//                << r.m_begin[1] << "), ("
//                << r.m_end[0] << ", "
//                << r.m_end[1] << "), "
//                << r.m_first << ", "
//                << r.m_last
//                << std::endl;
//            }
//        }
//        std::cout.flush();
//    }*/
//
//
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        //std::cout << grid_1;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        //std::cout << grid_2;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==1)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        //std::cout << grid_1;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        //std::cout << grid_2;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==2)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        //std::cout << grid_1;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        //std::cout << grid_2;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//
//    /*if (rank==0)
//    {
//        // visit edge neighbors (principal neighbors)
//        grid_2.visit_neighbors<0>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
//        std::cout << std::endl;
//        // visit vertex neighbors
//        grid_2.visit_neighbors<1>(11,[](int& data, int id) { std::cout << data << ", " << id << std::endl; });
//        std::cout.flush();
//    }*/
//
//
//    // pack send values
//    std::vector<std::vector<int>> s_send_package_1;
//    s_map_1.pack(s_send_package_1, sgrid_1);
//
//    std::vector<std::vector<int>> s_send_package_2;
//    s_map_2.pack(s_send_package_2, sgrid_2);
//
//    /*MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0)
//    {
//        std::cout << "send package 1" << std::endl;
//        for (const auto& v : s_send_package_1)
//        {
//            for (const auto& x : v)
//                std::cout << x << " ";
//            std::cout << std::endl;
//        }
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);*/
//
//    auto s_domain_id_to_rank = [](int domain_id) { return domain_id/2; };
//
//    std::vector<std::vector<int>> s_recv_package_1;
//    auto reqs_1 = s_map_1.exchange(s_send_package_1, s_recv_package_1, s_domain_id_to_rank);
//
//    std::vector<std::vector<int>> s_recv_package_2;
//    auto reqs_2 = s_map_2.exchange(s_send_package_2, s_recv_package_2, s_domain_id_to_rank);
//
//    
//
//    for (auto& req : reqs_1)
//    {
//            MPI_Status st;
//            MPI_Wait(&req, &st);
//    }
//    for (auto& req : reqs_2)
//    {
//            MPI_Status st;
//            MPI_Wait(&req, &st);
//    }
//
//    /*MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0)
//    {
//        std::cout << "recv package 1" << std::endl;
//        for (const auto& v : s_recv_package_1)
//        {
//            for (const auto& x : v)
//                std::cout << x << " ";
//            std::cout << std::endl;
//        }
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);*/
//    
//
//    // unpack
//    s_map_1.unpack(s_recv_package_1, sgrid_1);
//    s_map_2.unpack(s_recv_package_2, sgrid_2);
//
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==0)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==1)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//    if (rank==2)
//    {
//        std::cout << "rank = " << rank << std::endl;
//        std::cout << "  id = " << rank*2 << std::endl;
//        std::cout << sgrid_1;
//        std::cout << "  id = " << rank*2+1 << std::endl;
//        std::cout << sgrid_2;
//        std::cout.flush();
//    }
//    MPI_Barrier(MPI_COMM_WORLD);
//   

    MPI_Finalize();
    return 0;
}

