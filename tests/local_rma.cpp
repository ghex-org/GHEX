/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */


#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/threads/std_thread/primitives.hpp>
#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif

#include <ghex/structured/range.hpp>
#include <ghex/structured/bulk_communication_object.hpp>

#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>


//#include <iomanip>
//#include <thread>
//#include <future>
//#ifdef __CUDACC__
//#include <gridtools/common/cuda_util.hpp>
//#include <gridtools/common/host_device.hpp>
//#endif

#include <gtest/gtest.h>

#include <array>
#include <gridtools/common/array.hpp>

template<typename T, std::size_t N>
using array_type = gridtools::array<T,N>;

using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,3>;

using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,3>;

template<typename T, typename Arch, int... Is>
using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type, Is...>;
    
using T1 = double;
using T2 = float;
using T3 = int;
using TT1 = array_type<T1,3>;
using TT2 = array_type<T2,3>;
using TT3 = array_type<T3,3>;

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
                f(xl,yl,zl) = array_type<T,3>{(T)x,(T)y,(T)z};
            }
        }
    }
}

template<typename T, typename Domain, typename Halos, typename Periodic, typename Global, typename Field, typename Communicator>
bool test_values(const Domain& d, const Halos& halos, const Periodic& periodic, const Global& g_first, const Global& g_last,
        const Field& f, Communicator comm)
{
    bool passed = true;
    const int i = d.domain_id()%2;
    int rank, size;
    MPI_Comm_rank(comm,&rank);
    MPI_Comm_size(comm,&size);

    int xl = -halos[0];
    int hxl = halos[0];
    int hxr = halos[1];
    // hack begin: make it work with 1 rank (works with even number of ranks otherwise)
    if (i==0 && size==1)//comm.rank()%2 == 0 && comm.rank()+1 == comm.size())
    {
        xl = 0;
        hxl = 0;
    }
    if (i==1 && size==1)//comm.rank()%2 == 0 && comm.rank()+1 == comm.size())
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
            if (d.domain_id()<2 &&      y<d.first()[1] && !periodic[1]) continue;
            if (d.domain_id()>size-3 && y>d.last()[1]  && !periodic[1]) continue;
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
                    << i << "  " << rank << std::endl;
                }
            }
        }
    }
    return passed;
}

TEST(local_rma, ctor)
{
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    //using communicator_type = typename std::remove_reference_t<decltype(context)>::communicator_type;
    
    // local portion per domain
    //const std::array<int,3> local_ext{10,15,20};
    const std::array<int,3> local_ext{4,3,2};
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
    const std::array<int,3> g_last {local_ext[0]*4-1, ((context.size()-1)/2+1)*local_ext[1]-1, local_ext[2]-1};
    // maximum halo
    const std::array<int,3> offset{3,3,3};
    // local size including potential halos
    const std::array<int,3> local_ext_buffer{local_ext[0]+2*offset[0], local_ext[1]+2*offset[1], local_ext[2]+2*offset[2]};
    // maximum number of elements per local domain
    const int max_memory = local_ext_buffer[0]*local_ext_buffer[1]*local_ext_buffer[2];
    
    std::vector<TT1> field_1a_raw(max_memory);
    std::vector<TT1> field_1b_raw(max_memory);
    std::vector<TT2> field_2a_raw(max_memory);
    std::vector<TT2> field_2b_raw(max_memory);
    std::vector<TT3> field_3a_raw(max_memory);
    std::vector<TT3> field_3b_raw(max_memory);

    // add local domains
    std::vector<domain_descriptor_type> local_domains;
    local_domains.push_back( domain_descriptor_type{
        context.rank()*2,
        std::array<int,3>{ ((context.rank()%2)*2  )*local_ext[0],   (context.rank()/2  )*local_ext[1],                0},
        std::array<int,3>{ ((context.rank()%2)*2+1)*local_ext[0]-1, (context.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}});
    local_domains.push_back( domain_descriptor_type{
        context.rank()*2+1,
        std::array<int,3>{ ((context.rank()%2)*2+1)*local_ext[0],   (context.rank()/2  )*local_ext[1],             0},
        std::array<int,3>{ ((context.rank()%2)*2+2)*local_ext[0]-1, (context.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}});

    // halo generators
    std::array<int,6> halos{2,2,2,2,2,2};
    auto halo_gen = halo_generator_type(g_first, g_last, halos, periodic);

    // make patterns
    auto pattern = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains);
    
    //using pattern_type = decltype(pattern);
    
    // wrap raw fields
    auto field_1a = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[0], field_1a_raw.data(), offset, local_ext_buffer);
    auto field_1b = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[1], field_1b_raw.data(), offset, local_ext_buffer);
    auto field_2a = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[0], field_2a_raw.data(), offset, local_ext_buffer);
    auto field_2b = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[1], field_2b_raw.data(), offset, local_ext_buffer);
    auto field_3a = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[0], field_3a_raw.data(), offset, local_ext_buffer);
    auto field_3b = gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(local_domains[1], field_3b_raw.data(), offset, local_ext_buffer);

    // fill arrays
    fill_values<T1>(local_domains[0], field_1a);
    fill_values<T1>(local_domains[1], field_1b);
    fill_values<T2>(local_domains[0], field_2a);
    fill_values<T2>(local_domains[1], field_2b);
    fill_values<T3>(local_domains[0], field_3a);
    fill_values<T3>(local_domains[1], field_3b);

    auto ex = gridtools::ghex::make_bulk_co< gridtools::ghex::structured::remote_thread_range_generator > (
        context.get_communicator(context.get_token()),
        pattern,
        field_1a,
        field_1b,
        field_2a,
        field_2b,
        field_3a,
        field_3b);

    ex.exchange();
    ex.exchange();
    ex.exchange();
    
    bool passed =true;
    passed = passed && test_values<T1>(local_domains[0], halos, periodic, g_first, g_last, field_1a, context.mpi_comm());
    passed = passed && test_values<T1>(local_domains[1], halos, periodic, g_first, g_last, field_1b, context.mpi_comm());
    passed = passed && test_values<T2>(local_domains[0], halos, periodic, g_first, g_last, field_2a, context.mpi_comm());
    passed = passed && test_values<T2>(local_domains[1], halos, periodic, g_first, g_last, field_2b, context.mpi_comm());
    passed = passed && test_values<T3>(local_domains[0], halos, periodic, g_first, g_last, field_3a, context.mpi_comm());
    passed = passed && test_values<T3>(local_domains[1], halos, periodic, g_first, g_last, field_3b, context.mpi_comm());
    
    EXPECT_TRUE(passed);
}
