// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
#include "../include/simple_field.hpp"
#include "../include/structured_pattern.hpp"
#include "../include/communication_object_erased.hpp"
#include <boost/mpi/environment.hpp>
#include <array>
#include <iomanip>

#include <thread>
#include <future>

#define MULTI_THREADED_EXCHANGE
#define MULTI_THREADED_EXCHANGE_THREADS
//#define MULTI_THREADED_EXCHANGE_ASYNC_ASYNC
//#define MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED
//#define MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT

template<typename T, long unsigned N>
std::ostream& operator<<(std::ostream& os, const std::array<T,N>& arr)
{
    os << "(";
    for (unsigned int i=0; i<N-1; ++i) os << std::setw(2) << std::right << arr[i] << ",";
    os << std::setw(2) << std::right << arr[N-1] << ")";
    return os;
}


using domain_descriptor_type = gridtools::structured_domain_descriptor<int,3>;
template<typename T, typename Device, int... Is>
using field_descriptor_type  = gridtools::simple_field_wrapper<T,Device,domain_descriptor_type, Is...>;

bool test0(boost::mpi::communicator& mpi_comm)
{
    // need communicator to decompose domain
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};

    // local portion per domain
    //const std::array<int,3> local_ext{10,15,20};
    const std::array<int,3> local_ext{4,3,2};

    // decomposition: 4 domains in x-direction, 1 domain in z-direction, rest in y-direction
    //                each MPI rank owns two domains: either first or last two domains in x-direction
    //
    //          +---------> x 
    //          |
    //          |     +------<0>------+------<1>------+
    //          |     | +----+ +----+ | +----+ +----+ |
    //          v     | |  0 | |  1 | | |  2 | |  3 | |
    //                | +----+ +----+ | +----+ +----+ |
    //          y     +------<2>------+------<3>------+
    //                | +----+ +----+ | +----+ +----+ |
    //                | |  5 | |  6 | | |  7 | |  8 | |
    //                | +----+ +----+ | +----+ +----+ |
    //                +------<4>------+------<5>------+
    //                | +----+ +----+ | +----+ +----+ |
    //                | |  9 | | 10 | | | 11 | | 12 | |
    //                . .    . .    . . .    . .    . .
    //                . .    . .    . . .    . .    . .
    //


    // compute total domain
    const std::array<int,3> g_first{               0,                                    0,              0};
    const std::array<int,3> g_last {local_ext[0]*4-1, ((comm.size()-1)/2+1)*local_ext[1]-1, local_ext[2]-1};
    // maximum halo
    const std::array<int,3> offset{3,3,3};
    // local size including potential halos
    const std::array<int,3> local_ext_buffer{local_ext[0]+2*offset[0], local_ext[1]+2*offset[1], local_ext[2]+2*offset[2]};
    // maximum number of elements per local domain
    const int max_memory = local_ext_buffer[0]*local_ext_buffer[1]*local_ext_buffer[2];

    // allocate fields
    std::vector<double> field_1a_raw(max_memory);
    std::vector<double> field_1b_raw(max_memory);
    std::vector<int> field_2a_raw(max_memory);
    std::vector<int> field_2b_raw(max_memory);
    std::vector<std::array<int,3>> field_3a_raw(max_memory);
    std::vector<std::array<int,3>> field_3b_raw(max_memory);

    // add local domains
    std::vector<domain_descriptor_type> local_domains;
    local_domains.push_back( domain_descriptor_type{
        comm.rank()*2, 
        std::array<int,3>{ ((comm.rank()%2)*2  )*local_ext[0],   (comm.rank()/2  )*local_ext[1],                0},
        std::array<int,3>{ ((comm.rank()%2)*2+1)*local_ext[0]-1, (comm.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}});
    local_domains.push_back( domain_descriptor_type{
        comm.rank()*2+1,
        std::array<int,3>{ ((comm.rank()%2)*2+1)*local_ext[0],   (comm.rank()/2  )*local_ext[1],             0},
        std::array<int,3>{ ((comm.rank()%2)*2+2)*local_ext[0]-1, (comm.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}});

    // halo generators
    auto halo_gen1 = domain_descriptor_type::halo_generator_type(
        g_first, g_last,
        {1,1,1,1,1,1}, 
        {true,true,true});
    auto halo_gen2 = domain_descriptor_type::halo_generator_type(
        g_first, g_last,
        {2,2,2,2,2,2}, 
        {true,true,true});

    // make patterns
    auto pattern1 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen1, local_domains);
    auto pattern2 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen2, local_domains);
    
    // communication object
    auto co = gridtools::make_communication_object(pattern1,pattern2);
    auto co_1 = gridtools::make_communication_object(pattern1,pattern2);
    auto co_2 = gridtools::make_communication_object(pattern1,pattern2);
    
    // wrap raw fields
    field_descriptor_type<double, gridtools::device::cpu, 2, 1, 0> field_1a { 
        local_domains[0].domain_id(), field_1a_raw.data(), offset, local_ext_buffer};
    field_descriptor_type<double, gridtools::device::cpu, 2, 1, 0> field_1b { 
        local_domains[1].domain_id(), field_1b_raw.data(), offset, local_ext_buffer};
    
    field_descriptor_type<int, gridtools::device::cpu, 2, 1, 0> field_2a { 
        local_domains[0].domain_id(), field_2a_raw.data(), offset, local_ext_buffer};
    field_descriptor_type<int, gridtools::device::cpu, 2, 1, 0> field_2b { 
        local_domains[1].domain_id(), field_2b_raw.data(), offset, local_ext_buffer};
    
    field_descriptor_type<std::array<int,3>, gridtools::device::cpu, 2, 1, 0> field_3a { 
        local_domains[0].domain_id(), field_3a_raw.data(), offset, local_ext_buffer};
    field_descriptor_type<std::array<int,3>, gridtools::device::cpu, 2, 1, 0> field_3b { 
        local_domains[1].domain_id(), field_3b_raw.data(), offset, local_ext_buffer};

    // fill arrays
    { 
        int xl = 0;
        for (int x=local_domains[0].first()[0]; x<=local_domains[0].last()[0]; ++x, ++xl)
        { 
            int yl = 0;
            for (int y=local_domains[0].first()[1]; y<=local_domains[0].last()[1]; ++y, ++yl)
            { 
                int zl = 0;
                for (int z=local_domains[0].first()[2]; z<=local_domains[0].last()[2]; ++z, ++zl)
                {
                    field_3a(xl,yl,zl) = std::array<int,3>{x,y,z};
                }
            }
        }
    }
    {
        int xl = 0;
        for (int x=local_domains[1].first()[0]; x<=local_domains[1].last()[0]; ++x, ++xl)
        {
            int yl = 0;
            for (int y=local_domains[1].first()[1]; y<=local_domains[1].last()[1]; ++y, ++yl)
            {
                int zl = 0;
                for (int z=local_domains[1].first()[2]; z<=local_domains[1].last()[2]; ++z, ++zl)
                {
                    field_3b(xl,yl,zl) = std::array<int,3>{x,y,z};
                }
            }
        }
    }
    
    // print arrays
    std::cout.flush();
    comm.barrier();
    for (int r=0; r<comm.size(); ++r)
    {
        if (r!=comm.rank())
        {
            std::cout.flush();
            comm.barrier();
            continue;
        }
        std::cout << "rank " << r << std::endl;
        std::cout << std::endl;
        for (int z=-1; z<local_ext[2]+1; ++z)
        {
            std::cout << "z = " << z << std::endl; 
            std::cout << std::endl;
            for (int y=-1; y<local_ext[1]+1; ++y)
            {
                for (int x=-1; x<local_ext[0]+1; ++x)
                {
                    std::cout << field_3a(x,y,z) << " ";
                }
                std::cout << "      ";
                for (int x=-1; x<local_ext[0]+1; ++x)
                {
                    std::cout << field_3b(x,y,z) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout.flush();
        comm.barrier();
    }

    // exchange
#ifndef MULTI_THREADED_EXCHANGE
    co.bexchange(
        pattern1(field_1a),
        pattern1(field_1b),
        pattern2(field_2a),
        pattern2(field_2b),
        pattern1(field_3a),
        pattern1(field_3b)
    );
#else
    auto func = [](decltype(co)& co_, auto... bis) 
    { 
        co_.bexchange(bis...);
        /*auto h = co_.exchange(bis...);
        //std::this_thread::yield();
        h.wait();*/
    };
    auto func_h = [](decltype(co)& co_, auto... bis) 
    { 
        return co_.exchange(bis...);
    };
#ifdef MULTI_THREADED_EXCHANGE_THREADS
    std::vector<std::thread> threads;
    threads.push_back(std::thread{func, std::ref(co_1), 
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a)});
    threads.push_back(std::thread{func, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b)});
    for (auto& t : threads) t.join();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_ASYNC) 
    auto policy = std::launch::async;
    auto future_1 = std::async(policy, func, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    future_1.wait();
    future_2.wait();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED) 
    auto policy = std::launch::deferred;
    auto future_1 = std::async(policy, func_h, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func_h, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    auto h1 = future_1.get();
    auto h2 = future_2.get();
    h1.wait();
    h2.wait();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT) 
    auto policy = std::launch::async;
    auto future_1 = std::async(policy, func_h, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func_h, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    future_1.get().wait();
    future_2.get().wait();
#endif
#endif

    // print arrays
    std::cout.flush();
    comm.barrier();
    for (int r=0; r<comm.size(); ++r)
    {
        if (r!=comm.rank())
        {
            std::cout.flush();
            comm.barrier();
            continue;
        }
        std::cout << "rank " << r << std::endl;
        std::cout << std::endl;
        for (int z=-1; z<local_ext[2]+1; ++z)
        {
            std::cout << "z = " << z << std::endl; 
            std::cout << std::endl;
            for (int y=-1; y<local_ext[1]+1; ++y)
            {
                for (int x=-1; x<local_ext[0]+1; ++x)
                {
                    std::cout << field_3a(x,y,z) << " ";
                }
                std::cout << "      ";
                for (int x=-1; x<local_ext[0]+1; ++x)
                {
                    std::cout << field_3b(x,y,z) << " ";
                }
                std::cout << std::endl;
            }
            std::cout << std::endl;
        }
        std::cout.flush();
        comm.barrier();
    }

    return true;
}

int main(int argc, char* argv[])
{
    //MPI_Init(&argc,&argv);
#ifdef MULTI_THREADED_EXCHANGE
    int provided;
    int res = MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    if (res == MPI_ERR_OTHER)
    {
        throw std::runtime_error("MPI init failed");
    }
    if (provided < MPI_THREAD_MULTIPLE)
    {
        throw std::runtime_error("MPI does not support threading");
    }
#else
    boost::mpi::environment env(argc, argv);
#endif

    boost::mpi::communicator world;
    auto passed = test0(world);
    
#ifdef MULTI_THREADED_EXCHANGE
    MPI_Finalize();
#endif
    return 0;
}

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

