/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
//#define STANDALONE

//#define SERIAL_SPLIT
//#define MULTI_THREADED_EXCHANGE
//#define MULTI_THREADED_EXCHANGE_THREADS
//#define MULTI_THREADED_EXCHANGE_ASYNC_ASYNC
//#define MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED
//#define MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT

#include "../include/simple_field_wrapper.hpp"
#include "../include/structured_pattern.hpp"
#include "../include/communication_object_2.hpp"
#include <boost/mpi/environment.hpp>
#include <array>
#include <iomanip>

#include <thread>
#include <future>

#ifndef STANDALONE
#include <gtest/gtest.h>
#include "gtest_main.cpp"
#endif

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


template<typename T, typename Domain, typename Field>
void fill_values(const Domain& d, Field& f)
{ 
    int xl = 0;
    for (int x=d.first()[0]; x<=d.last()[0]; ++x, ++xl)
    { 
        int yl = 0;
        for (int y=d.first()[1]; y<=d.last()[1]; ++y, ++yl)
        { 
            int zl = 0;
            for (int z=d.first()[2]; z<=d.last()[2]; ++z, ++zl)
            {
                f(xl,yl,zl) = std::array<T,3>{(T)x,(T)y,(T)z};
            }
        }
    }
}

template<typename T, typename Domain, typename Halos, typename Periodic, typename Global, typename Field, typename Communicator>
bool test_values(const Domain& d, const Halos& halos, const Periodic& periodic, const Global& g_first, const Global& g_last, 
        const Field& f, Communicator& comm)
{
    bool passed = true;
    const int i = d.domain_id()%2;

    int xl = -halos[0];
    int hxl = halos[0];
    int hxr = halos[1];
    // hack begin: make it work with 1 rank (works with even number of ranks otherwise)
    if (i==0 && comm.size()==1)//comm.rank()%2 == 0 && comm.rank()+1 == comm.size()) 
    {
        xl = 0;
        hxl = 0;
    }
    if (i==1 && comm.size()==1)//comm.rank()%2 == 0 && comm.rank()+1 == comm.size()) 
    {
        hxr = 0;
    }
    // hack end
    for (int x=d.first()[0]-hxl; x<=d.last()[0]+hxr; ++x, ++xl)
    {
        if (i==0 && x<d.first()[0] && !periodic[0]) continue;
        if (i==1 && x>d.last()[0]  && !periodic[0]) continue;
        T x_wrapped = (((x-g_first[0])+(g_last[0]-g_first[0]+1))%(g_last[0]-g_first[0]+1) + g_first[0]);
        int yl = -halos[2];
        for (int y=d.first()[1]-halos[2]; y<=d.last()[1]+halos[3]; ++y, ++yl)
        { 
            if (d.domain_id()<2 &&             y<d.first()[1] && !periodic[1]) continue;
            if (d.domain_id()>comm.size()-3 && y>d.last()[1]  && !periodic[1]) continue;
            T y_wrapped = (((y-g_first[1])+(g_last[1]-g_first[1]+1))%(g_last[1]-g_first[1]+1) + g_first[1]);
            int zl = -halos[4];
            for (int z=d.first()[2]-halos[4]; z<=d.last()[2]+halos[5]; ++z, ++zl)
            {
                if (z<d.first()[2] && !periodic[2]) continue;
                if (z>d.last()[2]  && !periodic[2]) continue;
                T z_wrapped = (((z-g_first[2])+(g_last[2]-g_first[2]+1))%(g_last[2]-g_first[2]+1) + g_first[2]);

                const auto& value = f(xl,yl,zl);
                if(value[0]!=x_wrapped || value[1]!=y_wrapped || value[2]!=z_wrapped)
                {
                    passed = false;
                    std::cout 
                    << "(" << xl << ", " << yl << ", " << zl << ") values found != expected: " 
                    << "(" << value[0] << ", " << value[1] << ", " << value[2] << ") != "
                    << "(" << x_wrapped << ", " << y_wrapped << ", " << z_wrapped << ") " //<< std::endl;
                    << i << "  " << comm.rank() << std::endl;
                }
            }
        }
    }
    return passed;
}


#ifndef STANDALONE
TEST(communication_object_2, exchange)
#else
bool test0()
#endif
{
    boost::mpi::communicator mpi_comm;

    // need communicator to decompose domain
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};

    // local portion per domain
    const std::array<int,3> local_ext{10,15,20};
    //const std::array<int,3> local_ext{4,3,2};
    const std::array<bool,3> periodic{true,true,true};

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
    //                | |  4 | |  5 | | |  6 | |  7 | |
    //                | +----+ +----+ | +----+ +----+ |
    //                +------<4>------+------<5>------+
    //                | +----+ +----+ | +----+ +----+ |
    //                | |  8 | |  9 | | | 10 | | 11 | |
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
    using T1 = double;
    using T2 = float;
    using T3 = int;
    using TT1 = std::array<T1,3>;
    using TT2 = std::array<T2,3>;
    using TT3 = std::array<T3,3>;
    std::vector<TT1> field_1a_raw(max_memory);
    std::vector<TT1> field_1b_raw(max_memory);
    std::vector<TT2> field_2a_raw(max_memory);
    std::vector<TT2> field_2b_raw(max_memory);
    std::vector<TT3> field_3a_raw(max_memory);
    std::vector<TT3> field_3b_raw(max_memory);

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
    std::array<int,6> halos1{0,0,1,0,1,2};
    std::array<int,6> halos2{2,2,2,2,2,2};
    auto halo_gen1 = domain_descriptor_type::halo_generator_type(g_first, g_last, halos1, periodic); 
    auto halo_gen2 = domain_descriptor_type::halo_generator_type(g_first, g_last, halos2, periodic); 

    // make patterns
    auto pattern1 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen1, local_domains);
    auto pattern2 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen2, local_domains);
    
    // communication object
    auto co   = gridtools::make_communication_object(pattern1,pattern2);
    auto co_1 = gridtools::make_communication_object(pattern1,pattern2);
    auto co_2 = gridtools::make_communication_object(pattern1,pattern2);

    // wrap raw fields
    auto field_1a = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[0].domain_id(), field_1a_raw.data(), offset, local_ext_buffer);
    auto field_1b = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[1].domain_id(), field_1b_raw.data(), offset, local_ext_buffer);
    auto field_2a = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[0].domain_id(), field_2a_raw.data(), offset, local_ext_buffer);
    auto field_2b = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[1].domain_id(), field_2b_raw.data(), offset, local_ext_buffer);
    auto field_3a = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[0].domain_id(), field_3a_raw.data(), offset, local_ext_buffer);
    auto field_3b = gridtools::wrap_field<gridtools::device::cpu,2,1,0>(local_domains[1].domain_id(), field_3b_raw.data(), offset, local_ext_buffer);
    
    // fill arrays
    fill_values<T1>(local_domains[0], field_1a);
    fill_values<T1>(local_domains[1], field_1b);
    fill_values<T2>(local_domains[0], field_2a);
    fill_values<T2>(local_domains[1], field_2b);
    fill_values<T3>(local_domains[0], field_3a);
    fill_values<T3>(local_domains[1], field_3b);
    
    //// print arrays
    //std::cout.flush();
    //comm.barrier();
    //for (int r=0; r<comm.size(); ++r)
    //{
    //    if (r!=comm.rank())
    //    {
    //        std::cout.flush();
    //        comm.barrier();
    //        continue;
    //    }
    //    std::cout << "rank " << r << std::endl;
    //    std::cout << std::endl;
    //    for (int z=-1; z<local_ext[2]+1; ++z)
    //    {
    //        std::cout << "z = " << z << std::endl; 
    //        std::cout << std::endl;
    //        for (int y=-1; y<local_ext[1]+1; ++y)
    //        {
    //            for (int x=-1; x<local_ext[0]+1; ++x)
    //            {
    //                std::cout << field_3a(x,y,z) << " ";
    //            }
    //            std::cout << "      ";
    //            for (int x=-1; x<local_ext[0]+1; ++x)
    //            {
    //                std::cout << field_3b(x,y,z) << " ";
    //            }
    //            std::cout << std::endl;
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout.flush();
    //    comm.barrier();
    //}



    // exchange
#ifndef MULTI_THREADED_EXCHANGE
#ifndef SERIAL_SPLIT
    // blocking variant
    co.bexchange(
        pattern1(field_1a),
        pattern1(field_1b),
        pattern2(field_2a),
        pattern2(field_2b),
        pattern1(field_3a),
        pattern1(field_3b)
    );
#else
    // non-blocking variant
    auto h1 = co_1.exchange(pattern1(field_1a), pattern2(field_2a), pattern1(field_3a));
    auto h2 = co_2.exchange(pattern1(field_1b), pattern2(field_2b), pattern1(field_3b));
    // ... overlap communication (packing, posting) with computation here
    // wait and upack:
    h1.wait();
    h2.wait();
#endif
#else
#ifdef MULTI_THREADED_EXCHANGE_THREADS
    auto func = [](decltype(co)& co_, auto... bis) 
    { 
        co_.bexchange(bis...);
    };
    // packing and posting may be done concurrently
    // waiting and unpacking may be done concurrently
    std::vector<std::thread> threads;
    threads.push_back(std::thread{func, std::ref(co_1), 
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a)});
    threads.push_back(std::thread{func, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b)});
    // ... overlap communication with computation here
    for (auto& t : threads) t.join();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_ASYNC) 
    auto func = [](decltype(co)& co_, auto... bis) 
    { 
        co_.bexchange(bis...);
    };
    // packing and posting may be done concurrently
    // waiting and unpacking may be done concurrently
    auto policy = std::launch::async;
    auto future_1 = std::async(policy, func, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    // ... overlap communication with computation here
    future_1.wait();
    future_2.wait();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_DEFERRED) 
    auto func_h = [](decltype(co)& co_, auto... bis) 
    { 
        return co_.exchange(bis...);
    };
    // packing and posting serially on current thread
    // waiting and unpacking serially on current thread
    auto policy = std::launch::deferred;
    auto future_1 = std::async(policy, func_h, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func_h, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    // deferred policy: essentially serial on current thread
    auto h1 = future_1.get();
    auto h2 = future_2.get();
    // ... overlap communication (packing, posting) with computation here
    // waiting and unpacking is serial here
    h1.wait();
    h2.wait();
#elif defined(MULTI_THREADED_EXCHANGE_ASYNC_ASYNC_WAIT) 
    auto func_h = [](decltype(co)& co_, auto... bis) 
    { 
        return co_.exchange(bis...);
    };
    // packing and posting may be done concurrently
    // waiting and unpacking serially
    auto policy = std::launch::async;
    auto future_1 = std::async(policy, func_h, std::ref(co_1),
        pattern1(field_1a), 
        pattern2(field_2a), 
        pattern1(field_3a));
    auto future_2 = std::async(policy, func_h, std::ref(co_2),
        pattern1(field_1b), 
        pattern2(field_2b), 
        pattern1(field_3b));
    // ... overlap communication (packing, posting) with computation here
    // waiting and unpacking is serial here
    future_1.get().wait();
    future_2.get().wait();
#endif
#endif

    //// print arrays
    //std::cout.flush();
    //comm.barrier();
    //for (int r=0; r<comm.size(); ++r)
    //{
    //    if (r!=comm.rank())
    //    {
    //        std::cout.flush();
    //        comm.barrier();
    //        continue;
    //    }
    //    std::cout << "rank " << r << std::endl;
    //    std::cout << std::endl;
    //    for (int z=-1; z<local_ext[2]+1; ++z)
    //    {
    //        std::cout << "z = " << z << std::endl; 
    //        std::cout << std::endl;
    //        for (int y=-1; y<local_ext[1]+1; ++y)
    //        {
    //            for (int x=-1; x<local_ext[0]+1; ++x)
    //            {
    //                std::cout << field_3a(x,y,z) << " ";
    //            }
    //            std::cout << "      ";
    //            for (int x=-1; x<local_ext[0]+1; ++x)
    //            {
    //                std::cout << field_3b(x,y,z) << " ";
    //            }
    //            std::cout << std::endl;
    //        }
    //        std::cout << std::endl;
    //    }
    //    std::cout.flush();
    //    comm.barrier();
    //}

    bool passed =true;
    passed = passed && test_values<T1>(local_domains[0], halos1, periodic, g_first, g_last, field_1a, comm);
    passed = passed && test_values<T1>(local_domains[1], halos1, periodic, g_first, g_last, field_1b, comm);
    passed = passed && test_values<T2>(local_domains[0], halos2, periodic, g_first, g_last, field_2a, comm);
    passed = passed && test_values<T2>(local_domains[1], halos2, periodic, g_first, g_last, field_2b, comm);
    passed = passed && test_values<T3>(local_domains[0], halos1, periodic, g_first, g_last, field_3a, comm);
    passed = passed && test_values<T3>(local_domains[1], halos1, periodic, g_first, g_last, field_3b, comm);
    
#ifdef STANDALONE
    if (passed)
        std::cout << "RESULT: PASSED" << std::endl;
    else
        std::cout << "RESULT: FAILED" << std::endl;
    return passed;
#else
    EXPECT_TRUE(passed);
#endif
}

#ifdef STANDALONE
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

    auto passed = test0();
    
#ifdef MULTI_THREADED_EXCHANGE
    MPI_Finalize();
#endif
    return 0;
}
#endif

