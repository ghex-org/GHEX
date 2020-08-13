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

#include <iostream>
#include <thread>
#include <mutex>
#include <chrono>
#include <pthread.h>
#include <array>
#include <memory>
#include <gridtools/common/array.hpp>
#include <omp.h>

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

#include <ghex/structured/remote_range.hpp>
#include <ghex/structured/bulk_communication_object.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>

using clock_type = std::chrono::high_resolution_clock;

struct simulation
{
#ifdef __CUDACC__
    template<typename T>
    struct cuda_deleter
    {
        void operator()(T* ptr)
        {
            cudaFree(ptr);
        }
    };
#endif

    using T = GHEX_FLOAT_TYPE;

    using context_type = gridtools::ghex::tl::context<transport,threading>;
    using context_ptr_type = std::unique_ptr<context_type>;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,3>;
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,3>;
    template<typename Arch, int... Is>
    using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type, Is...>;

    using field_type     = field_descriptor_type<gridtools::ghex::cpu, 2, 1, 0>;
#ifdef __CUDACC__
    using gpu_field_type = field_descriptor_type<gridtools::ghex::gpu, 2, 1, 0>;
#endif

    int num_reps;
    int num_threads;
    bool mt;
    const int num_fields;
    int ext;
    context_ptr_type context_ptr;
    context_type& context;
    const std::array<int,3> local_ext;
    const std::array<bool,3> periodic;
    const std::array<int,3> g_first;
    const std::array<int,3> g_last;
    const std::array<int,3> offset;
    const std::array<int,3> local_ext_buffer;
    
    std::array<int,6> halos;
    halo_generator_type halo_gen;

    std::array<int,3> p_decomp;
    std::array<int,3> t_decomp;
    std::vector<domain_descriptor_type> local_domains;

    const int max_memory;
    std::vector<std::vector<std::vector<T>>> fields_raw;
    std::vector<std::vector<field_type>> fields;
#ifdef __CUDACC__
    std::vector<std::vector<std::unique_ptr<T,cuda_deleter<T>>>> fields_raw_gpu;
    std::vector<std::vector<gpu_field_type>> fields_gpu;
#endif

    typename context_type::communicator_type comm;
    std::vector<typename context_type::communicator_type> comms;
    std::vector<gridtools::ghex::bulk_communication_object> cos;

    using pattern_type = std::remove_reference_t<decltype(
        gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains))>;
    std::unique_ptr<pattern_type> pattern;

    std::mutex io_mutex;

    simulation(
        int num_reps_,
        int ext_,
        int halo,
        std::array<int,3> pd,
        std::array<int,3> td,
        int num_threads_)
    : num_reps{num_reps_}
    , num_threads(num_threads_)
    , mt(num_threads > 1)
    , num_fields{8}//{num_fields_}
    , ext{ext_}
    , context_ptr{gridtools::ghex::tl::context_factory<transport,threading>::create(num_threads, MPI_COMM_WORLD)}
    , context{*context_ptr}
    , local_ext{ext,ext,ext}
    , periodic{true,true,true}
    , g_first{0,0,0}
    , g_last{local_ext[0]*pd[0]*td[0]-1,local_ext[1]*pd[1]*td[1]-1,local_ext[2]*pd[2]*td[2]-1}
    , offset{halo,halo,halo}
    , local_ext_buffer{local_ext[0]+2*offset[0], local_ext[1]+2*offset[1], local_ext[2]+2*offset[2]}
    , halos{halo,halo,halo,halo,halo,halo}
    , halo_gen(g_first, g_last, halos, periodic)
    , p_decomp{pd}
    , t_decomp{td}
    , max_memory{local_ext_buffer[0]*local_ext_buffer[1]*local_ext_buffer[2]}
    , comm{ context.get_serial_communicator() }
    {
        // compute decomposition
        int pz = comm.rank() / (pd[0]*pd[1]);
        int py = (comm.rank() - pz*pd[0]*pd[1]) / pd[0];
        int px = comm.rank() - pz*pd[0]*pd[1] - py*pd[0];
        
        local_domains.reserve(num_threads);
        fields_raw.reserve(num_threads);
        fields.reserve(num_threads);
#ifdef __CUDACC__
        fields_raw_gpu.reserve(num_threads);
        fields_gpu.reserve(num_threads);
#endif
        comms.reserve(num_threads);
        omp_set_num_threads(num_threads);
#pragma omp parallel for ordered schedule(static,1)
        for (int j=0; j<num_threads; ++j)
        {
            int tz = j / (td[0]*td[1]);
            int ty = (j - tz*td[0]*td[1]) / td[0];
            int tx = j - tz*td[0]*td[1] - ty*td[0];
            int x = (px*td[0] + tx)*ext;
            int y = (py*td[1] + ty)*ext;
            int z = (pz*td[2] + tz)*ext;
#pragma omp ordered
            {
                local_domains.push_back(domain_descriptor_type{
                    context.rank()*num_threads+j,
                    std::array<int,3>{x,y,z},
                    std::array<int,3>{x+ext-1,y+ext-1,z+ext-1}});
                fields_raw.resize(fields_raw.size()+1);
                fields.resize(fields.size()+1);
#ifdef __CUDACC__
                fields_raw_gpu.resize(fields_raw_gpu.size()+1);
                fields_gpu.resize(fields_gpu.size()+1);
#endif
                for (int i=0; i<num_fields; ++i)
                {
                    fields_raw.back().push_back( std::vector<T>(max_memory) );
                    fields.back().push_back(
                        gridtools::ghex::wrap_field<gridtools::ghex::cpu,2,1,0>(
                            local_domains.back(),
                            fields_raw.back().back().data(),
                            offset,
                            local_ext_buffer));
#ifdef __CUDACC__
                    fields_raw_gpu.back().push_back( 
                        std::unique_ptr<T,cuda_deleter<T>>{
                            [this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(T)); return (T*)ptr; }()});
                    fields_gpu.back().push_back(
                        gridtools::ghex::wrap_field<gridtools::ghex::gpu,2,1,0>(
                            local_domains.back(),
                            fields_raw_gpu.back().back().get(),
                            offset,
                            local_ext_buffer));
#endif
                }
                comms.push_back(context.get_communicator(context.get_token()));
            }
        }

        pattern = std::unique_ptr<pattern_type>{
            new pattern_type{
                gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(
                        context, halo_gen, local_domains)}};
        
        cos.resize(num_threads);
#pragma omp parallel
        {
            const auto j = omp_get_thread_num();

            auto bco = gridtools::ghex::structured::bulk_communication_object<
                gridtools::ghex::structured::remote_range_generator,
                pattern_type,
                field_type
            > (comms[j], *pattern);

#ifndef __CUDACC__
            for (int i=0; i<num_fields; ++i)
                bco.add_field(fields[j][i]);
#else
            for (int i=0; i<num_fields; ++i)
                bco.add_field(fields_gpu[j][i]);
#endif
            
            cos[j] = std::move(bco);
        }
    }

    void exchange()
    {
#pragma omp parallel
        {

            const auto j = omp_get_thread_num();
                std::this_thread::sleep_for(std::chrono::milliseconds(50));
                {
                    std::lock_guard<std::mutex> io_lock(io_mutex);
                    std::cout << "Thread #" << j << ": on CPU " << sched_getcpu() << "\n";
                }
    
                // warm up
                for (int t = 0; t < 50; ++t)
                {
                    cos[j].exchange().wait();
                }
    
                auto start = clock_type::now();
                for (int t = 0; t < num_reps; ++t)
                {
                    cos[j].exchange().wait();
                }
                auto end = clock_type::now();
                std::chrono::duration<double> elapsed_seconds = end - start;
                
                if (comm.rank() == 0 && j == 0)
                {
                    const auto num_elements = 
                        (ext+halos[0]+halos[1]) * (ext+halos[2]+halos[3]) * (ext+halos[2]+halos[3]) -
                        ext * ext * ext;
                    const auto   num_bytes = num_elements * sizeof(T);
                    const double load = 2 * comm.size() * num_threads * num_fields * num_bytes;
                    const auto   GB_per_s = num_reps * load / (elapsed_seconds.count() * 1.0e9);
                    std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
                    std::cout << "GB/s : " << GB_per_s << std::endl;
                }
        }
    }
};

void print_usage(const char* app_name)
{
    std::cout
        << "<mpi-launcher> -np N " << app_name << " "
        << "local-domain-size "
        << "num-repetition "
        << "halo-size "
        << "process-domain-decompositon "
        << "thread-domain-decompositon "
        << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 10)
    {
        print_usage(argv[0]);
        return 1;
    }

    int domain_size = std::atoi(argv[1]);
    int num_repetitions = std::atoi(argv[2]);
    int halo = std::atoi(argv[3]);
    std::array<int,3> proc_decomposition;
    int num_ranks = 1;
    for (int i = 4; i < 7; ++i)
    {
        proc_decomposition[i - 4] = std::atoi(argv[i]);
        num_ranks *= proc_decomposition[i-4];
    }
    std::array<int,3> thread_decomposition;
    int num_threads = 1;
    for (int i = 7; i < 10; ++i)
    {
        thread_decomposition[i - 7] = std::atoi(argv[i]);
        num_threads *= thread_decomposition[i-7];
    }
    
    int required = num_threads>1 ?  MPI_THREAD_MULTIPLE :  MPI_THREAD_SINGLE;
    int provided;
    int init_result = MPI_Init_thread(&argc, &argv, required, &provided);
    if (init_result == MPI_ERR_OTHER)
    {
        std::cerr << "MPI init failed\n";
        std::terminate();
    }
    if (provided < required)
    {
        std::cerr << "MPI does not support required threading level\n";
        std::terminate();
    }

    MPI_Barrier(MPI_COMM_WORLD);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD,&world_size);
    if (world_size != num_ranks)
    {
        std::cout << "processor decomposition is wrong" << std::endl;
        print_usage(argv[0]);
        return 1;
    }

    {
    simulation sim(num_repetitions, domain_size, halo, proc_decomposition, thread_decomposition, num_threads);

    sim.exchange();
    
    MPI_Barrier(MPI_COMM_WORLD);

    }
    MPI_Finalize();

    return 0;
}
