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

#include <gtest/gtest.h>
#include <thread>
#include <array>
#include <gridtools/common/array.hpp>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#else
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#endif

#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <ghex/common/cuda_runtime.hpp>
#endif



struct simulation_1
{
#ifdef GHEX_CUDACC
    template<typename T>
    struct cuda_deleter
    {
        void operator()(T* ptr)
        {
            cudaFree(ptr);
        }
    };
#endif

    template<typename T, std::size_t N>
    using array_type = gridtools::array<T,N>;
    using T1 = double;
    using T2 = float;
    using T3 = int;
    using TT1 = array_type<T1,3>;
    using TT2 = array_type<T2,3>;
    using TT3 = array_type<T3,3>;

    using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
    using context_ptr_type = std::unique_ptr<context_type>;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,std::integral_constant<int, 3>>;
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,std::integral_constant<int, 3>>;
    template<typename T, typename Arch, int... Is>
    using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type, ::gridtools::layout_map<Is...>>;

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

    context_ptr_type context_ptr;
    context_type& context;
    const std::array<int,3> local_ext;
    const std::array<bool,3> periodic;
    const std::array<int,3> g_first;
    const std::array<int,3> g_last;
    const std::array<int,3> offset;
    const std::array<int,3> local_ext_buffer;
    const int max_memory;
    std::vector<TT1> field_1a_raw;
    std::vector<TT1> field_1b_raw;
    std::vector<TT2> field_2a_raw;
    std::vector<TT2> field_2b_raw;
    std::vector<TT3> field_3a_raw;
    std::vector<TT3> field_3b_raw;
#ifdef GHEX_CUDACC
    std::unique_ptr<TT1,cuda_deleter<TT1>> field_1a_raw_gpu;
    std::unique_ptr<TT1,cuda_deleter<TT1>> field_1b_raw_gpu;
    std::unique_ptr<TT2,cuda_deleter<TT2>> field_2a_raw_gpu;
    std::unique_ptr<TT2,cuda_deleter<TT2>> field_2b_raw_gpu;
    std::unique_ptr<TT3,cuda_deleter<TT3>> field_3a_raw_gpu;
    std::unique_ptr<TT3,cuda_deleter<TT3>> field_3b_raw_gpu;
#endif /* GHEX_CUDACC */
    std::vector<domain_descriptor_type> local_domains;
    std::array<int,6> halos;
    halo_generator_type halo_gen;
    using pattern_type = std::remove_reference_t<decltype(
        gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains))>;
    pattern_type pattern;
    field_descriptor_type<TT1, gridtools::ghex::cpu, 2, 1, 0> field_1a;
    field_descriptor_type<TT1, gridtools::ghex::cpu, 2, 1, 0> field_1b;
    field_descriptor_type<TT2, gridtools::ghex::cpu, 2, 1, 0> field_2a;
    field_descriptor_type<TT2, gridtools::ghex::cpu, 2, 1, 0> field_2b;
    field_descriptor_type<TT3, gridtools::ghex::cpu, 2, 1, 0> field_3a;
    field_descriptor_type<TT3, gridtools::ghex::cpu, 2, 1, 0> field_3b;
#ifdef GHEX_CUDACC
    field_descriptor_type<TT1, gridtools::ghex::gpu, 2, 1, 0> field_1a_gpu;
    field_descriptor_type<TT1, gridtools::ghex::gpu, 2, 1, 0> field_1b_gpu;
    field_descriptor_type<TT2, gridtools::ghex::gpu, 2, 1, 0> field_2a_gpu;
    field_descriptor_type<TT2, gridtools::ghex::gpu, 2, 1, 0> field_2b_gpu;
    field_descriptor_type<TT3, gridtools::ghex::gpu, 2, 1, 0> field_3a_gpu;
    field_descriptor_type<TT3, gridtools::ghex::gpu, 2, 1, 0> field_3b_gpu;
#endif /* GHEX_CUDACC */
    typename context_type::communicator_type comm;
    std::vector<typename context_type::communicator_type> comms;
    bool mt;
    using co_type = decltype(gridtools::ghex::make_communication_object<pattern_type>(comm));
    std::vector<co_type> basic_cos;
    std::vector<gridtools::ghex::generic_bulk_communication_object> cos;


    simulation_1(bool multithread = false)
    : context_ptr{ gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD) }
    , context{*context_ptr}
    , local_ext{4,3,2}
    , periodic{true,true,true}
    , g_first{0,0,0}
    , g_last{local_ext[0]*4-1, ((context.size()-1)/2+1)*local_ext[1]-1, local_ext[2]-1}
    , offset{3,3,3}
    , local_ext_buffer{local_ext[0]+2*offset[0], local_ext[1]+2*offset[1], local_ext[2]+2*offset[2]}
    , max_memory{local_ext_buffer[0]*local_ext_buffer[1]*local_ext_buffer[2]}
    , field_1a_raw(max_memory)
    , field_1b_raw(max_memory)
    , field_2a_raw(max_memory)
    , field_2b_raw(max_memory)
    , field_3a_raw(max_memory)
    , field_3b_raw(max_memory)
#ifdef GHEX_CUDACC
    , field_1a_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT1)); return (TT1*)ptr; }())
    , field_1b_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT1)); return (TT1*)ptr; }())
    , field_2a_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT2)); return (TT2*)ptr; }())
    , field_2b_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT2)); return (TT2*)ptr; }())
    , field_3a_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT3)); return (TT3*)ptr; }())
    , field_3b_raw_gpu([this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(TT3)); return (TT3*)ptr; }())
#endif /* GHEX_CUDACC */
    , local_domains{
        domain_descriptor_type{
            context.rank()*2,
            std::array<int,3>{ ((context.rank()%2)*2  )*local_ext[0],   (context.rank()/2  )*local_ext[1],                0},
            std::array<int,3>{ ((context.rank()%2)*2+1)*local_ext[0]-1, (context.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}},
        domain_descriptor_type{
            context.rank()*2+1,
            std::array<int,3>{ ((context.rank()%2)*2+1)*local_ext[0],   (context.rank()/2  )*local_ext[1],             0},
            std::array<int,3>{ ((context.rank()%2)*2+2)*local_ext[0]-1, (context.rank()/2+1)*local_ext[1]-1, local_ext[2]-1}}}
    , halos{2,2,2,2,2,2}
    , halo_gen(g_first, g_last, halos, periodic)
    , pattern{gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains)}
    , field_1a{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_1a_raw.data(), offset, local_ext_buffer)}
    , field_1b{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_1b_raw.data(), offset, local_ext_buffer)}
    , field_2a{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_2a_raw.data(), offset, local_ext_buffer)}
    , field_2b{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_2b_raw.data(), offset, local_ext_buffer)}
    , field_3a{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_3a_raw.data(), offset, local_ext_buffer)}
    , field_3b{gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_3b_raw.data(), offset, local_ext_buffer)}
#ifdef GHEX_CUDACC
    , field_1a_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_1a_raw_gpu.get(), offset, local_ext_buffer)}
    , field_1b_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_1b_raw_gpu.get(), offset, local_ext_buffer)}
    , field_2a_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_2a_raw_gpu.get(), offset, local_ext_buffer)}
    , field_2b_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_2b_raw_gpu.get(), offset, local_ext_buffer)}
    , field_3a_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[0], field_3a_raw_gpu.get(), offset, local_ext_buffer)}
    , field_3b_gpu{gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(local_domains[1], field_3b_raw_gpu.get(), offset, local_ext_buffer)}
#endif /* GHEX_CUDACC */
    , comm{ context.get_serial_communicator() }
    , mt{multithread}
    {
        fill_values(local_domains[0], field_1a);
        fill_values(local_domains[1], field_1b);
        fill_values(local_domains[0], field_2a);
        fill_values(local_domains[1], field_2b);
        fill_values(local_domains[0], field_3a);
        fill_values(local_domains[1], field_3b);
#ifdef GHEX_CUDACC
         cudaMemcpy(field_1a_raw_gpu.get(), field_1a_raw.data(), max_memory*sizeof(TT1), cudaMemcpyHostToDevice);
         cudaMemcpy(field_1b_raw_gpu.get(), field_1b_raw.data(), max_memory*sizeof(TT1), cudaMemcpyHostToDevice);
         cudaMemcpy(field_2a_raw_gpu.get(), field_2a_raw.data(), max_memory*sizeof(TT2), cudaMemcpyHostToDevice);
         cudaMemcpy(field_2b_raw_gpu.get(), field_2b_raw.data(), max_memory*sizeof(TT2), cudaMemcpyHostToDevice);
         cudaMemcpy(field_3a_raw_gpu.get(), field_3a_raw.data(), max_memory*sizeof(TT3), cudaMemcpyHostToDevice);
         cudaMemcpy(field_3b_raw_gpu.get(), field_3b_raw.data(), max_memory*sizeof(TT3), cudaMemcpyHostToDevice);
#endif /* GHEX_CUDACC */

        if (!mt)
        {
            comms.push_back(context.get_communicator());
            basic_cos.push_back(gridtools::ghex::make_communication_object<pattern_type>(comms[0]));
#ifndef GHEX_CUDACC
            auto bco =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::cpu, 2, 1, 0>
            > (basic_cos[0]);

            bco.add_field(pattern(field_1a));
            bco.add_field(pattern(field_1b));
            bco.add_field(pattern(field_2a));
            bco.add_field(pattern(field_2b));
            bco.add_field(pattern(field_3a));
            bco.add_field(pattern(field_3b));
#else
            auto bco =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::gpu, 2, 1, 0>
            > (basic_cos[0]);

            bco.add_field(pattern(field_1a_gpu));
            bco.add_field(pattern(field_1b_gpu));
            bco.add_field(pattern(field_2a_gpu));
            bco.add_field(pattern(field_2b_gpu));
            bco.add_field(pattern(field_3a_gpu));
            bco.add_field(pattern(field_3b_gpu));
#endif
            cos.emplace_back( std::move(bco) );

        } else {
            comms.push_back(context.get_communicator());
            comms.push_back(context.get_communicator());
            basic_cos.push_back(gridtools::ghex::make_communication_object<pattern_type>(comms[0]));
            basic_cos.push_back(gridtools::ghex::make_communication_object<pattern_type>(comms[1]));
#ifndef GHEX_CUDACC
            auto bco0 =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::cpu, 2, 1, 0>
            > (basic_cos[0]);
            bco0.add_field(pattern(field_1a));
            bco0.add_field(pattern(field_2a));
            bco0.add_field(pattern(field_3a));

            auto bco1 =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::cpu, 2, 1, 0>
            > (basic_cos[1]);
            bco1.add_field(pattern(field_1b));
            bco1.add_field(pattern(field_2b));
            bco1.add_field(pattern(field_3b));
#else
            auto bco0 =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::gpu, 2, 1, 0>
            > (basic_cos[0]);
            bco0.add_field(pattern(field_1a_gpu));
            bco0.add_field(pattern(field_2a_gpu));
            bco0.add_field(pattern(field_3a_gpu));

            auto bco1 =  gridtools::ghex::bulk_communication_object<
                gridtools::ghex::structured::rma_range_generator,
                pattern_type,
                field_descriptor_type<TT1, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, gridtools::ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, gridtools::ghex::gpu, 2, 1, 0>
            > (basic_cos[1]);
            bco1.add_field(pattern(field_1b_gpu));
            bco1.add_field(pattern(field_2b_gpu));
            bco1.add_field(pattern(field_3b_gpu));
#endif
            cos.emplace_back( std::move(bco0) );
            cos.emplace_back( std::move(bco1) );
        }
    }

    void exchange()
    {
        if (!mt)
        {
            cos[0].exchange().wait();
        }
        else
        {
            std::vector<std::thread> threads;
            threads.push_back(std::thread{[this]() -> void { cos[0].exchange().wait(); }});
            threads.push_back(std::thread{[this]() -> void { cos[1].exchange().wait(); }});
            for (auto& t : threads) t.join();
        }
    }

    bool check()
    {
#ifdef GHEX_CUDACC
         cudaMemcpy(field_1a_raw.data(), field_1a_raw_gpu.get(), max_memory*sizeof(TT1), cudaMemcpyDeviceToHost);
         cudaMemcpy(field_1b_raw.data(), field_1b_raw_gpu.get(), max_memory*sizeof(TT1), cudaMemcpyDeviceToHost);
         cudaMemcpy(field_2a_raw.data(), field_2a_raw_gpu.get(), max_memory*sizeof(TT2), cudaMemcpyDeviceToHost);
         cudaMemcpy(field_2b_raw.data(), field_2b_raw_gpu.get(), max_memory*sizeof(TT2), cudaMemcpyDeviceToHost);
         cudaMemcpy(field_3a_raw.data(), field_3a_raw_gpu.get(), max_memory*sizeof(TT3), cudaMemcpyDeviceToHost);
         cudaMemcpy(field_3b_raw.data(), field_3b_raw_gpu.get(), max_memory*sizeof(TT3), cudaMemcpyDeviceToHost);
#endif /* GHEX_CUDACC */
        bool passed =true;
        passed = passed && test_values(local_domains[0], field_1a);
        passed = passed && test_values(local_domains[1], field_1b);
        passed = passed && test_values(local_domains[0], field_2a);
        passed = passed && test_values(local_domains[1], field_2b);
        passed = passed && test_values(local_domains[0], field_3a);
        passed = passed && test_values(local_domains[1], field_3b);
        return passed;
    }

private:
    template<typename Field>
    void fill_values(const domain_descriptor_type& d, Field& f)
    {
        using T = typename Field::value_type::value_type;
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

    template<typename Field>
    bool test_values(const domain_descriptor_type& d, const Field& f)
    {
        using T = typename Field::value_type::value_type;
        bool passed = true;
        const int i = d.domain_id()%2;
        int rank = comm.rank();
        int size = comm.size();

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
};

TEST(local_rma, single)
{
    simulation_1 sim(false);
    sim.exchange();
    sim.exchange();
    sim.exchange();
    EXPECT_TRUE(sim.check());
}

TEST(local_rma, multi)
{
    simulation_1 sim(true);
    sim.exchange();
    sim.exchange();
    sim.exchange();
    EXPECT_TRUE(sim.check());
}
