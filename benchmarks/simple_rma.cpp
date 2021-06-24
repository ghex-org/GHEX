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
#include <iomanip>
#include <thread>
#include <mutex>
#include <chrono>
#include <pthread.h>
#include <array>
#include <memory>
#include <gridtools/common/array.hpp>
#include <pthread.h>
#include <thread>
#include <vector>

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
#include <ghex/util/decomposition.hpp>
#include <ghex/common/timer.hpp>

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <ghex/common/cuda_runtime.hpp>
#endif

using clock_type = std::chrono::high_resolution_clock;

struct simulation
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

    using T = GHEX_FLOAT_TYPE;

    using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
    using context_ptr_type = std::unique_ptr<context_type>;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,std::integral_constant<int, 3>>;
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,std::integral_constant<int, 3>>;
    template<typename Arch, typename Layout>
    using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type,Layout>;

    using field_type     = field_descriptor_type<gridtools::ghex::cpu, ::gridtools::layout_map<2, 1, 0>>;
#ifdef GHEX_CUDACC
    using gpu_field_type = field_descriptor_type<gridtools::ghex::gpu, ::gridtools::layout_map<2, 1, 0>>;
#endif
    using decomp_type = gridtools::ghex::hierarchical_decomposition<3>;

    int num_reps;
    decomp_type decomp;
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
    std::array<int,6> halos;
    const std::array<int,3> local_ext_buffer;
    halo_generator_type halo_gen;
    std::vector<domain_descriptor_type> local_domains;
    const int max_memory;
    std::vector<std::vector<std::vector<T>>> fields_raw;
    std::vector<std::vector<field_type>> fields;
#ifdef GHEX_CUDACC
    std::vector<std::vector<std::unique_ptr<T,cuda_deleter<T>>>> fields_raw_gpu;
    std::vector<std::vector<gpu_field_type>> fields_gpu;
#endif
    typename context_type::communicator_type comm;
    std::vector<typename context_type::communicator_type> comms;
    std::vector<gridtools::ghex::generic_bulk_communication_object> cos;

    using pattern_type = std::remove_reference_t<decltype(
        gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen, local_domains))>;
    std::unique_ptr<pattern_type> pattern;
    std::mutex io_mutex;

    std::vector<gridtools::ghex::timer> timer_vec;

    simulation(
        int num_reps_,
        int ext_,
        int halo,
        int num_fields_,
        const decomp_type& decomp_)
    : num_reps{num_reps_}
    , decomp(decomp_)
    , num_threads(decomp.threads_per_rank())
    , mt(num_threads > 1)
    , num_fields{num_fields_}
    , ext{ext_}
    , context_ptr{gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD)}
    , context{*context_ptr}
    , local_ext{ext,ext,ext}
    , periodic{true,true,true}
    , g_first{0,0,0}
    , g_last{
        decomp.last_coord()[0]*local_ext[0]+local_ext[0]-1,
        decomp.last_coord()[1]*local_ext[1]+local_ext[1]-1,
        decomp.last_coord()[2]*local_ext[2]+local_ext[2]-1}
    , offset{halo,halo,halo}
    , halos{halo,halo,halo,halo,halo,halo}
    , local_ext_buffer{
        local_ext[0]+halos[0]+halos[1],
        local_ext[1]+halos[2]+halos[3],
        local_ext[2]+halos[4]+halos[5]}
    , halo_gen(g_first, g_last, halos, periodic)
    , max_memory{local_ext_buffer[0]*local_ext_buffer[1]*local_ext_buffer[2]}
    , comm{ context.get_serial_communicator() }
    , timer_vec(num_threads)
    {
        cos.resize(num_threads);
        local_domains.reserve(num_threads);
        fields_raw.resize(num_threads);
        fields.resize(num_threads);
#ifdef GHEX_CUDACC
        fields_raw_gpu.resize(num_threads);
        fields_gpu.resize(num_threads);
#endif
        comms = std::vector<typename context_type::communicator_type>(num_threads, comm);

        for (int j=0; j<num_threads; ++j)
        {
            const auto coord = decomp(comm.rank(), j);
            int x = coord[0]*local_ext[0];
            int y = coord[1]*local_ext[1];
            int z = coord[2]*local_ext[2];
            local_domains.push_back(domain_descriptor_type{
                context.rank()*num_threads+j,
                std::array<int,3>{x,y,z},
                std::array<int,3>{x+local_ext[0]-1,y+local_ext[1]-1,z+local_ext[2]-1}});
        }

        pattern = std::unique_ptr<pattern_type>{new pattern_type{
            gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(
                context, halo_gen, local_domains)}};
    }

    void exchange()
    {
        if (num_threads == 1)
        {
            std::thread t([this](){exchange(0);});
            // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
            // only CPU = local rank as set.
            cpu_set_t cpuset;
            CPU_ZERO(&cpuset);
            CPU_SET(decomp.node_resource(comm.rank()), &cpuset);
            int rc = pthread_setaffinity_np(t.native_handle(), sizeof(cpu_set_t), &cpuset);
            if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
            t.join();
        }
        else
        {
            std::vector<std::thread> threads;
            threads.reserve(num_threads);
            for (int j=0; j<num_threads; ++j)
            {
                threads.push_back( std::thread([this,j](){ exchange(j); }) );
                // Create a cpu_set_t object representing a set of CPUs. Clear it and mark
                // only CPU j as set.
                cpu_set_t cpuset;
                CPU_ZERO(&cpuset);
                CPU_SET(decomp.node_resource(comm.rank(),j), &cpuset);
                int rc = pthread_setaffinity_np(threads[j].native_handle(), sizeof(cpu_set_t), &cpuset);
                if (rc != 0) { std::cerr << "Error calling pthread_setaffinity_np: " << rc << "\n"; }
            }
            for (auto& t : threads) t.join();
        }
    }

    void exchange(int j)
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        comms[j] = context.get_communicator();
        auto basic_co = gridtools::ghex::make_communication_object<pattern_type>(comms[j]);
        for (int i=0; i<num_fields; ++i)
        {
            fields_raw[j].push_back( std::vector<T>(max_memory) );
            fields[j].push_back(gridtools::ghex::wrap_field<gridtools::ghex::cpu,::gridtools::layout_map<2,1,0>>(
                local_domains[j],
                fields_raw[j].back().data(),
                offset,
                local_ext_buffer));
#ifdef GHEX_CUDACC
            fields_raw_gpu[j].push_back( std::unique_ptr<T,cuda_deleter<T>>{
                [this](){ void* ptr; cudaMalloc(&ptr, max_memory*sizeof(T)); return (T*)ptr; }()});
            fields_gpu[j].push_back(gridtools::ghex::wrap_field<gridtools::ghex::gpu,::gridtools::layout_map<2,1,0>>(
                local_domains[j],
                fields_raw_gpu[j].back().get(),
                offset,
                local_ext_buffer));
#endif
        }

        auto bco = gridtools::ghex::bulk_communication_object<
            gridtools::ghex::structured::rma_range_generator,
            pattern_type,
#ifndef GHEX_CUDACC
            field_type
#else
            gpu_field_type
#endif
        > (basic_co);
#ifndef GHEX_CUDACC
        for (int i=0; i<num_fields; ++i)
            bco.add_field(pattern->operator()(fields[j][i]));
#else
        for (int i=0; i<num_fields; ++i)
            bco.add_field(pattern->operator()(fields_gpu[j][i]));
#endif
        cos[j] = std::move(bco);

        // warm up
        for (int t = 0; t < 50; ++t)
        {
            cos[j].exchange().wait();
        }

        auto start = clock_type::now();
        for (int t = 0; t < num_reps; ++t)
        {
            timer_vec[j].tic();
            cos[j].exchange().wait();
            timer_vec[j].toc();
            std::cout << "mean time:    " << comm.rank() << ":" << j << " " << std::setprecision(12) << timer_vec[j].mean()/1000000.0 << "\n";
            timer_vec[j].clear();
        }
        auto end = clock_type::now();
        std::chrono::duration<double> elapsed_seconds = end - start;

        if (comm.rank() == 0 && j == 0)
        {
            const auto num_elements =
                local_ext_buffer[0] * local_ext_buffer[1] * local_ext_buffer[2]
                - local_ext[0] * local_ext[1] * local_ext[2];
            const auto   num_bytes = num_elements * sizeof(T);
            const double load = 2 * comm.size() * num_threads * num_fields * num_bytes;
            const auto   GB_per_s = num_reps * load / (elapsed_seconds.count() * 1.0e9);
            std::cout << "elapsed time: " << elapsed_seconds.count() << "s\n";
            std::cout << "GB/s : " << GB_per_s << std::endl;
            const auto tt = timer_vec[0];
            std::cout << "mean time:    " << std::setprecision(12) << tt.mean()/1000000.0 << "\n";
            std::cout << "min time:     " << std::setprecision(12) << tt.min()/1000000.0 << "\n";
            std::cout << "max time:     " << std::setprecision(12) << tt.max()/1000000.0 << "\n";
            std::cout << "sdev time:    " << std::setprecision(12) << tt.stddev()/1000000.0 << "\n";
            std::cout << "sdev f time:  " << std::setprecision(12) << tt.stddev()/tt.mean() << "\n";
            std::cout << "GB/s mean:    " << std::setprecision(12) << load / (tt.mean()*1000.0) << std::endl;
            std::cout << "GB/s min:     " << std::setprecision(12) << load / (tt.max()*1000.0) << std::endl;
            std::cout << "GB/s max:     " << std::setprecision(12) << load / (tt.min()*1000.0) << std::endl;
            std::cout << "GB/s sdev:    " << std::setprecision(12) << (tt.stddev()/tt.mean())* (load / (tt.mean()*1000.0)) << std::endl;
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
        << "num-fields "
        << "node-decompositon "
        << "numa-decompositon "
        << "rank-decompositon "
        << "thread-decompositon "
        << std::endl;
}

int main(int argc, char** argv)
{
    if (argc != 17)
    {
        print_usage(argv[0]);
        return 1;
    }

    int domain_size = std::atoi(argv[1]);
    int num_repetitions = std::atoi(argv[2]);
    int halo = std::atoi(argv[3]);
    int num_fields = std::atoi(argv[4]);
    std::array<int,3> node_decomposition;
    std::array<int,3> numa_decomposition;
    std::array<int,3> rank_decomposition;
    std::array<int,3> thread_decomposition;
    int num_ranks = 1;
    int num_threads = 1;
    for (int i = 0; i < 3; ++i)
    {
        node_decomposition[i] = std::atoi(argv[i+5]);
        numa_decomposition[i] = std::atoi(argv[i+5+3]);
        rank_decomposition[i] = std::atoi(argv[i+5+6]);
        thread_decomposition[i] = std::atoi(argv[i+5+9]);
        num_ranks *= node_decomposition[i]*numa_decomposition[i]*rank_decomposition[i];
        num_threads *= thread_decomposition[i];
    }

    typename simulation::decomp_type decomp(node_decomposition, numa_decomposition,
        rank_decomposition, thread_decomposition);

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
        simulation sim(num_repetitions, domain_size, halo, num_fields, decomp);

        sim.exchange();
    
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();

    return 0;
}
