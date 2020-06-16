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

#ifndef STANDALONE
#include "gtest/gtest.h"
#endif
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>
#include <array>

#include "../utils/triplet.hpp"

#include <ghex/communication_object_2.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/transport_layer/mpi/context.hpp>
#include <ghex/threads/atomic/primitives.hpp>
#include <ghex/common/timer.hpp>

#include <gridtools/common/array.hpp>
#ifdef __CUDACC__
#include <gridtools/common/cuda_util.hpp>
#include <gridtools/common/host_device.hpp>
#endif

using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::atomic::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

namespace halo_exchange_3D_generic_full {

    using timer_type = gridtools::ghex::timer;


    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

#define B_ADD 1
#define C_ADD 2

#ifdef VECTOR_INTERFACE
    typedef int T1;
    typedef int T2;
    typedef int T3;
#else
    typedef int T1;
    typedef double T2;
    typedef long long int T3;
#endif

    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<int,3>;
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<int,3>;
    template<typename T, typename Arch, int... Is>
    using field_descriptor_type  = gridtools::ghex::structured::regular::field_descriptor<T,Arch,domain_descriptor_type, Is...>;

#ifdef __CUDACC__
    using arch_type = gridtools::ghex::gpu;
#else
    using arch_type = gridtools::ghex::cpu;
#endif

    template<typename T, typename Arch, typename DomainDescriptor, int... Order>
    void printbuff(std::ostream& file, const gridtools::ghex::structured::regular::field_descriptor<T,Arch,DomainDescriptor, Order...>& field)
    {
        if (field.extents()[0] <= 10 && field.extents()[1] <= 10 && field.extents()[2] <= 6)
        {
            file << "------------\n";
            for (int kk = 0; kk < field.extents()[2]; ++kk) {
                for (int jj = 0; jj < field.extents()[1]; ++jj) {
                    file << "|";
                    for (int ii = 0; ii < field.extents()[0]; ++ii) {
                        file << field(ii-field.offsets()[0], jj-field.offsets()[1], kk-field.offsets()[2]);
                    }
                    file << "|\n";
                }
                file << "\n\n";
            }
            file << "------------\n\n";
        }
    }

    template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2, typename Comm>
    bool run(ST &file, context_type& context, Comm comm,
        int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3,
        triple_t<USE_DOUBLE, T1> *_a,
        triple_t<USE_DOUBLE, T2> *_b,
        triple_t<USE_DOUBLE, T3> *_c, bool use_gpu) 
    {
        // compute total domain
        const std::array<int,3> g_first{             0,              0,              0};
        const std::array<int,3> g_last {dims[0]*DIM1-1, dims[1]*DIM2-1, dims[2]*DIM3-1};

        // periodicity
        const std::array<bool,3> periodic{per0,per1,per2};

        // halos
        const std::array<int,6> halo_1{H1m1,H1p1,H2m1,H2p1,H3m1,H3p1};
#ifndef GHEX_1_PATTERN_BENCHMARK
        const std::array<int,6> halo_2{H1m2,H1p2,H2m2,H2p2,H3m2,H3p2};
        const std::array<int,6> halo_3{H1m3,H1p3,H2m3,H2p3,H3m3,H3p3};
#endif

        // define local domain
        domain_descriptor_type local_domain{
            context.rank(),//comm.rank(),
            std::array<int,3>{coords[0]*DIM1,coords[1]*DIM2,coords[2]*DIM3},
            std::array<int,3>{(coords[0]+1)*DIM1-1,(coords[1]+1)*DIM2-1,(coords[2]+1)*DIM3-1}};
        std::vector<domain_descriptor_type> local_domains{local_domain};

        // wrap raw fields
        auto a = gridtools::ghex::wrap_field<gridtools::ghex::cpu,I1,I2,I3>(local_domain, _a,
            std::array<int,3>{H1m1,H2m1,H3m1},
            std::array<int,3>{(DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1)});
        auto b = gridtools::ghex::wrap_field<gridtools::ghex::cpu,I1,I2,I3>(local_domain, _b,
            std::array<int,3>{H1m2,H2m2,H3m2},
            std::array<int,3>{(DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2)});
        auto c = gridtools::ghex::wrap_field<gridtools::ghex::cpu,I1,I2,I3>(local_domain, _c,
            std::array<int,3>{H1m3,H2m3,H3m3},
            std::array<int,3>{(DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3)});

        // make halo generators
        auto halo_gen_1 = halo_generator_type(g_first, g_last, halo_1, periodic);
#ifndef GHEX_1_PATTERN_BENCHMARK
        auto halo_gen_2 = halo_generator_type(g_first, g_last, halo_2, periodic);
        auto halo_gen_3 = halo_generator_type(g_first, g_last, halo_3, periodic);
#endif

        // make patterns
        auto pattern_1 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen_1, local_domains);
#ifndef GHEX_1_PATTERN_BENCHMARK
        auto pattern_2 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen_2, local_domains);
        auto pattern_3 = gridtools::ghex::make_pattern<gridtools::ghex::structured::grid>(context, halo_gen_3, local_domains);
#endif
        // communication object
        auto co = gridtools::ghex::make_communication_object<decltype(pattern_1)>(comm);


        file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";

        /* Just an initialization */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj)
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk)
                    a(ii-H1m1, jj-H2m1, kk-H3m1) = triple_t<USE_DOUBLE, T1>();

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj)
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk)
                    b(ii-H1m2, jj-H2m2, kk-H3m2) = triple_t<USE_DOUBLE, T2>();

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj)
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk)
                    c(ii-H1m3, jj-H2m3, kk-H3m3) = triple_t<USE_DOUBLE, T3>();

        for (int ii = 0; ii < DIM1; ++ii)
            for (int jj = 0; jj < DIM2; ++jj)
                for (int kk = 0; kk < DIM3; ++kk)
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
                        ii + (DIM1)*coords[0], jj + (DIM2)*coords[1], kk + (DIM3)*coords[2]);

        for (int ii = 0; ii < DIM1; ++ii)
            for (int jj = 0; jj < DIM2; ++jj)
                for (int kk = 0; kk < DIM3; ++kk)
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>(
                        ii + (DIM1)*coords[0] + B_ADD, jj + (DIM2)*coords[1] + B_ADD, kk + (DIM3)*coords[2] + B_ADD);

        for (int ii = 0; ii < DIM1; ++ii)
            for (int jj = 0; jj < DIM2; ++jj)
                for (int kk = 0; kk < DIM3; ++kk)
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>(
                        ii + (DIM1)*coords[0] + C_ADD, jj + (DIM2)*coords[1] + C_ADD, kk + (DIM3)*coords[2] + C_ADD);

        file << "A \n";
        printbuff(file, a);
        file << "B \n";
        printbuff(file, b);
        file << "C \n";
        printbuff(file, c);
        file.flush();

        if (use_gpu)
        {
            triple_t<USE_DOUBLE, T1>::data_type *gpu_a = 0;
            triple_t<USE_DOUBLE, T2>::data_type *gpu_b = 0;
            triple_t<USE_DOUBLE, T3>::data_type *gpu_c = 0;
            file << "***** GPU ON *****\n";

#ifdef __CUDACC__
            GT_CUDA_CHECK(cudaMalloc(&gpu_a,
                (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) *
                    sizeof(triple_t<USE_DOUBLE, T1>::data_type)));
            GT_CUDA_CHECK(cudaMalloc(&gpu_b,
                (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) *
                    sizeof(triple_t<USE_DOUBLE, T2>::data_type)));
            GT_CUDA_CHECK(cudaMalloc(&gpu_c,
                (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) *
                    sizeof(triple_t<USE_DOUBLE, T3>::data_type)));

            GT_CUDA_CHECK(cudaMemcpy(gpu_a,
                a.data(),
                (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) *
                    sizeof(triple_t<USE_DOUBLE, T1>::data_type),
                cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMemcpy(gpu_b,
                b.data(),
                (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) *
                    sizeof(triple_t<USE_DOUBLE, T2>::data_type),
                cudaMemcpyHostToDevice));

            GT_CUDA_CHECK(cudaMemcpy(gpu_c,
                c.data(),
                (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) *
                    sizeof(triple_t<USE_DOUBLE, T3>::data_type),
                cudaMemcpyHostToDevice));
#else
            gpu_a = new triple_t<USE_DOUBLE, T1>[(DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1)];
            gpu_b = new triple_t<USE_DOUBLE, T2>[(DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2)];
            gpu_c = new triple_t<USE_DOUBLE, T3>[(DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3)];

            std::memcpy((void*)gpu_a, (const void*)a.data(), 
                (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) * sizeof(triple_t<USE_DOUBLE, T1>::data_type));
            std::memcpy((void*)gpu_b, (const void*)b.data(), 
                (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) * sizeof(triple_t<USE_DOUBLE, T2>::data_type));
            std::memcpy((void*)gpu_c, (const void*)c.data(), 
                (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) * sizeof(triple_t<USE_DOUBLE, T3>::data_type));
#endif

            // wrap raw fields
            auto field1 = gridtools::ghex::wrap_field<arch_type,I1,I2,I3>(local_domain, gpu_a,
                std::array<int,3>{H1m1,H2m1,H3m1},
                std::array<int,3>{(DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1)});
            auto field2 = gridtools::ghex::wrap_field<arch_type,I1,I2,I3>(local_domain, gpu_b,
                std::array<int,3>{H1m2,H2m2,H3m2},
                std::array<int,3>{(DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2)});
            auto field3 = gridtools::ghex::wrap_field<arch_type,I1,I2,I3>(local_domain, gpu_c,
                std::array<int,3>{H1m3,H2m3,H3m3},
                std::array<int,3>{(DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3)});

            MPI_Barrier(context.mpi_comm());

            // do all the stuff here
            file << "                         LOCAL        MEAN          STD         MIN         MAX" << std::endl;
            timer_type t_0_local;
            timer_type t_1_local;
            timer_type t_local;
            timer_type t_0_global;
            timer_type t_1_global;
            timer_type t_global;
            const int k_start = 5;
            for (int k=0; k<25; ++k)
            {
                timer_type t_0;
                timer_type t_1;
                MPI_Barrier(context.mpi_comm());
                t_0.tic();
                auto h = co.exchange(
#ifndef GHEX_1_PATTERN_BENCHMARK
                    pattern_1(field1),
                    pattern_2(field2),
                    pattern_3(field3));
#else
                    pattern_1(field1),
                    pattern_1(field2),
                    pattern_1(field3));
#endif
                t_0.toc();
                t_1.tic();
                h.wait();
                t_1.toc();
                MPI_Barrier(context.mpi_comm());

                timer_type t;
                t(t_0.sum()+t_1.sum());

                auto t_0_all = gridtools::ghex::reduce(t_0,context.mpi_comm());
                auto t_1_all = gridtools::ghex::reduce(t_1,context.mpi_comm());
                auto t_all = gridtools::ghex::reduce(t,context.mpi_comm());
                if (k >= k_start)
                {
                    t_0_local(t_0);
                    t_1_local(t_1);
                    t_local(t);
                    t_0_global(t_0_all);
                    t_1_global(t_1_all);
                    t_global(t_all);
                }

                file << "TIME PACK/POST:   " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.mean()/1000.0
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_0_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.max()/1000.0
                    << std::endl;
                file << "TIME WAIT/UNPACK: " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.mean()/1000.0 
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_1_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.max()/1000.0
                    << std::endl;
                file << "TIME ALL:         " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.mean()/1000.0 
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.max()/1000.0
                    << std::endl;
                file << std::endl;
            }

            file << std::endl << "-----------------" << std::endl;
            file << "TIME PACK/POST:   " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_0_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.max()/1000.0
                << std::endl;
            file << "TIME WAIT/UNPACK: " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_1_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.max()/1000.0
                << std::endl;
            file << "TIME ALL:         " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.max()/1000.0
                << std::endl;

#ifdef __CUDACC__
            GT_CUDA_CHECK(cudaMemcpy(a.data(),
                gpu_a,
                (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) *
                    sizeof(triple_t<USE_DOUBLE, T1>::data_type),
                cudaMemcpyDeviceToHost));

            GT_CUDA_CHECK(cudaMemcpy(b.data(),
                gpu_b,
                (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) *
                    sizeof(triple_t<USE_DOUBLE, T2>::data_type),
                cudaMemcpyDeviceToHost));

            GT_CUDA_CHECK(cudaMemcpy(c.data(),
                gpu_c,
                (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) *
                    sizeof(triple_t<USE_DOUBLE, T3>::data_type),
                cudaMemcpyDeviceToHost));

            GT_CUDA_CHECK(cudaFree(gpu_a));
            GT_CUDA_CHECK(cudaFree(gpu_b));
            GT_CUDA_CHECK(cudaFree(gpu_c));
#else
            std::memcpy((void*)a.data(), (const void*)gpu_a, 
                (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) * sizeof(triple_t<USE_DOUBLE, T1>::data_type));
            std::memcpy((void*)b.data(), (const void*)gpu_b, 
                (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) * sizeof(triple_t<USE_DOUBLE, T2>::data_type));
            std::memcpy((void*)c.data(), (const void*)gpu_c, 
                (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) * sizeof(triple_t<USE_DOUBLE, T3>::data_type));
            
            delete[] gpu_a;
            delete[] gpu_b;
            delete[] gpu_c;
#endif

            MPI_Barrier(context.mpi_comm());

        }
        else
        {
            auto field1 = a;
            auto field2 = b;
            auto field3 = c;
            MPI_Barrier(context.mpi_comm());

            file << "                         LOCAL        MEAN          STD         MIN         MAX" << std::endl;
            timer_type t_0_local;
            timer_type t_1_local;
            timer_type t_local;
            timer_type t_0_global;
            timer_type t_1_global;
            timer_type t_global;
            const int k_start = 5;
            for (int k=0; k<25; ++k)
            {
                timer_type t_0;
                timer_type t_1;
                MPI_Barrier(context.mpi_comm());
                t_0.tic();
                auto h = co.exchange(
#ifndef GHEX_1_PATTERN_BENCHMARK
                    pattern_1(field1),
                    pattern_2(field2),
                    pattern_3(field3));
#else
                    pattern_1(field1),
                    pattern_1(field2),
                    pattern_1(field3));
#endif
                t_0.toc();
                t_1.tic();
                h.wait();
                t_1.toc();
                MPI_Barrier(context.mpi_comm());

                timer_type t;
                t(t_0.sum()+t_1.sum());

                auto t_0_all = gridtools::ghex::reduce(t_0,context.mpi_comm());
                auto t_1_all = gridtools::ghex::reduce(t_1,context.mpi_comm());
                auto t_all = gridtools::ghex::reduce(t,context.mpi_comm());
                if (k >= k_start)
                {
                    t_0_local(t_0);
                    t_1_local(t_1);
                    t_local(t);
                    t_0_global(t_0_all);
                    t_1_global(t_1_all);
                    t_global(t_all);
                }

                file << "TIME PACK/POST:   " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.mean()/1000.0
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_0_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_all.max()/1000.0
                    << std::endl;
                file << "TIME WAIT/UNPACK: " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.mean()/1000.0 
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_1_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_all.max()/1000.0
                    << std::endl;
                file << "TIME ALL:         " 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t.mean()/1000.0 
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.mean()/1000.0 
                    << " ±"
                    << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_all.stddev()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.min()/1000.0
                    << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_all.max()/1000.0
                    << std::endl;
                file << std::endl;
            }

            file << std::endl << "-----------------" << std::endl;
            file << "TIME PACK/POST:   " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_0_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_0_global.max()/1000.0
                << std::endl;
            file << "TIME WAIT/UNPACK: " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_1_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_1_global.max()/1000.0
                << std::endl;
            file << "TIME ALL:         " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_local.mean()/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.mean()/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << t_global.stddev()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.min()/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << t_global.max()/1000.0
                << std::endl;
            //file << std::endl << std::endl;

            MPI_Barrier(context.mpi_comm());
        }
        

        file << "\n********************************************************************************\n";

        file << "A \n";
        printbuff(file, a);
        file << "B \n";
        printbuff(file, b);
        file << "C \n";
        printbuff(file, c);
        file.flush();

        bool passed = true;


        /* Checking the data arrived correctly in the whole region
         */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj)
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {

                    triple_t<USE_DOUBLE, T1> ta;
                    int tax, tay, taz;

                    tax = modulus(ii - H1m1 + (DIM1)*coords[0], DIM1 * dims[0]);

                    tay = modulus(jj - H2m1 + (DIM2)*coords[1], DIM2 * dims[1]);

                    taz = modulus(kk - H3m1 + (DIM3)*coords[2], DIM3 * dims[2]);

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m1)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m1))) {
                            tax = triple_t<USE_DOUBLE, T1>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m1)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m1))) {
                            tay = triple_t<USE_DOUBLE, T1>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m1)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m1))) {
                            taz = triple_t<USE_DOUBLE, T1>().z();
                        }
                    }

                    ta = triple_t<USE_DOUBLE, T1>(tax, tay, taz).floor();

                    if (a(ii-H1m1, jj-H2m1, kk-H3m1) != ta) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "a " << a(ii-H1m1, jj-H2m1, kk-H3m1) << " != " << ta << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj)
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {

                    triple_t<USE_DOUBLE, T2> tb;
                    int tbx, tby, tbz;

                    tbx = modulus(ii - H1m2 + (DIM1)*coords[0], DIM1 * dims[0]) + B_ADD;

                    tby = modulus(jj - H2m2 + (DIM2)*coords[1], DIM2 * dims[1]) + B_ADD;

                    tbz = modulus(kk - H3m2 + (DIM3)*coords[2], DIM3 * dims[2]) + B_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m2)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m2))) {
                            tbx = triple_t<USE_DOUBLE, T2>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m2)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m2))) {
                            tby = triple_t<USE_DOUBLE, T2>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m2)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m2))) {
                            tbz = triple_t<USE_DOUBLE, T2>().z();
                        }
                    }

                    tb = triple_t<USE_DOUBLE, T2>(tbx, tby, tbz).floor();

                    if (b(ii-H1m2, jj-H2m2, kk-H3m2) != tb) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "b " << b(ii-H1m2, jj-H2m2, kk-H3m2) << " != " << tb << "\n";
                    }
                }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj)
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {

                    triple_t<USE_DOUBLE, T3> tc;
                    int tcx, tcy, tcz;

                    tcx = modulus(ii - H1m3 + (DIM1)*coords[0], DIM1 * dims[0]) + C_ADD;

                    tcy = modulus(jj - H2m3 + (DIM2)*coords[1], DIM2 * dims[1]) + C_ADD;

                    tcz = modulus(kk - H3m3 + (DIM3)*coords[2], DIM3 * dims[2]) + C_ADD;

                    if (!per0) {
                        if (((coords[0] == 0) && (ii < H1m3)) || ((coords[0] == dims[0] - 1) && (ii >= DIM1 + H1m3))) {
                            tcx = triple_t<USE_DOUBLE, T3>().x();
                        }
                    }

                    if (!per1) {
                        if (((coords[1] == 0) && (jj < H2m3)) || ((coords[1] == dims[1] - 1) && (jj >= DIM2 + H2m3))) {
                            tcy = triple_t<USE_DOUBLE, T3>().y();
                        }
                    }

                    if (!per2) {
                        if (((coords[2] == 0) && (kk < H3m3)) || ((coords[2] == dims[2] - 1) && (kk >= DIM3 + H3m3))) {
                            tcz = triple_t<USE_DOUBLE, T3>().z();
                        }
                    }

                    tc = triple_t<USE_DOUBLE, T3>(tcx, tcy, tcz).floor();

                    if (c(ii-H1m3, jj-H2m3, kk-H3m3) != tc) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "c " << c(ii-H1m3, jj-H2m3, kk-H3m3) << " != " << tc << "\n";
                    }
                }

        if (passed)
            file << "RESULT: PASSED!\n";
        else
            file << "RESULT: FAILED!\n";

        return passed;
    }
    
    bool test(bool use_gpu,
        int DIM1,
        int DIM2,
        int DIM3,
        int H1m1,
        int H1p1,
        int H2m1,
        int H2p1,
        int H3m1,
        int H3p1,
        int H1m2,
        int H1p2,
        int H2m2,
        int H2p2,
        int H3m2,
        int H3p2,
        int H1m3,
        int H1p3,
        int H2m3,
        int H2p3,
        int H3m3,
        int H3p3)
    {
        gridtools::ghex::tl::mpi::communicator_base world;
        //std::cout << context.rank() << " " << context.world().size() << "\n";

        std::stringstream ss;
        ss << world.rank();
        std::string filename = "comm_2_out" + ss.str() + ".txt";
        //std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << world.rank() << "  " << world.size() << "\n";
        dims[2] = 1;
        MPI_Dims_create(world.size(), 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << world.rank() << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

        MPI_Cart_create(world, 3, dims, period, false, &CartComm);

        MPI_Cart_get(CartComm, 3, dims, period, coords);
        
        auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, CartComm);
        auto& context = *context_ptr;
        auto comm = context.get_communicator(context.get_token());

        /* Each process will hold a tile of size
           (DIM1+2*H)x(DIM2+2*H)x(DIM3+2*H). The DIM1xDIM2xDIM3 area inside
           the H width border is the inner region of an hypothetical stencil
           computation whise halo width is H.
        */

        file << "Field A "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m1 << " - " << H1p1 << ", "
             << "Halo along j " << H2m1 << " - " << H2p1 << ", "
             << "Halo along k " << H3m1 << " - " << H3p1 << std::endl;

        file << "Field B "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m2 << " - " << H1p2 << ", "
             << "Halo along j " << H2m2 << " - " << H2p2 << ", "
             << "Halo along k " << H3m2 << " - " << H3p2 << std::endl;

        file << "Field C "
             << "size = " << DIM1 << "x" << DIM2 << "x" << DIM3 << " "
             << "Halo along i " << H1m3 << " - " << H1p3 << ", "
             << "Halo along j " << H2m3 << " - " << H2p3 << ", "
             << "Halo along k " << H3m3 << " - " << H3p3 << std::endl;
        file.flush();

        /* This example will exchange 3 data arrays at the same time with
           different values.
        */
        triple_t<USE_DOUBLE, T1> *_a =
            new triple_t<USE_DOUBLE, T1>[(DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1)];
        triple_t<USE_DOUBLE, T2> *_b =
            new triple_t<USE_DOUBLE, T2>[(DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2)];
        triple_t<USE_DOUBLE, T3> *_c =
            new triple_t<USE_DOUBLE, T3>[(DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3)];

        bool passed = true;

        file << "Permutation 0,1,2\n";

        file << "run<std::ostream, 0,1,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";

        passed = passed && run<std::ostream, 0, 1, 2, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 1, 2, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        file << "Permutation 0,2,1\n";

        file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 0, 2, 1, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,0,2\n";

        file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 0, 2, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,2,0\n";

        file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H31, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 1, 2, 0, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,0,1\n";

        file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 0, 1, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,1,0\n";

        file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, true, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, true, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, true, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, true, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, false, true, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, false, true, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file
            << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, false, false, true>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);

        file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed && run<std::ostream, 2, 1, 0, false, false, false>(file, context, comm,
                                DIM1,
                                DIM2,
                                DIM3,
                                H1m1,
                                H1p1,
                                H2m1,
                                H2p1,
                                H3m1,
                                H3p1,
                                H1m2,
                                H1p2,
                                H2m2,
                                H2p2,
                                H3m2,
                                H3p2,
                                H1m3,
                                H1p3,
                                H2m3,
                                H2p3,
                                H3m3,
                                H3p3,
                                _a,
                                _b,
                                _c, use_gpu);
        file << "---------------------------------------------------\n";

        delete[] _a;
        delete[] _b;
        delete[] _c;

        return passed;
    }

} // namespace halo_exchange_3D_generic_full

#ifdef STANDALONE
int main(int argc, char **argv)
{
#ifdef GT_USE_GPU
    device_binding();
#endif

#ifdef GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI
    int required = MPI_THREAD_MULTIPLE;
    int provided;
    int init_result = MPI_Init_thread(&argc, &argv, required, &provided);
    if (init_result == MPI_ERR_OTHER)
        throw std::runtime_error("MPI init failed");
    if (provided < required)
        throw std::runtime_error("MPI does not support required threading level");
#else
    MPI_Init(&argc,&argv);
#endif

    if (argc != 22) {
        std::cout << "Usage: test_halo_exchange_3D dimx dimy dimz h1m1 hip1 h2m1 h2m1 h3m1 h3p1 h1m2 hip2 h2m2 h2m2 "
                     "h3m2 h3p2 h1m3 hip3 h2m3 h2m3 h3m3 h3p3\n where args are integer sizes of the data fields and "
                     "halo width"
                  << std::endl;
        return 1;
    }
    int DIM1 = atoi(argv[1]);
    int DIM2 = atoi(argv[2]);
    int DIM3 = atoi(argv[3]);
    int H1m1 = atoi(argv[4]);
    int H1p1 = atoi(argv[5]);
    int H2m1 = atoi(argv[6]);
    int H2p1 = atoi(argv[7]);
    int H3m1 = atoi(argv[8]);
    int H3p1 = atoi(argv[9]);
    int H1m2 = atoi(argv[10]);
    int H1p2 = atoi(argv[11]);
    int H2m2 = atoi(argv[12]);
    int H2p2 = atoi(argv[13]);
    int H3m2 = atoi(argv[14]);
    int H3p2 = atoi(argv[15]);
    int H1m3 = atoi(argv[16]);
    int H1p3 = atoi(argv[17]);
    int H2m3 = atoi(argv[18]);
    int H2p3 = atoi(argv[19]);
    int H3m3 = atoi(argv[20]);
    int H3p3 = atoi(argv[21]);

    halo_exchange_3D_generic_full::test(DIM1,
        DIM2,
        DIM3,
        H1m1,
        H1p1,
        H2m1,
        H2p1,
        H3m1,
        H3p1,
        H1m2,
        H1p2,
        H2m2,
        H2p2,
        H3m2,
        H3p2,
        H1m3,
        H1p3,
        H2m3,
        H2p3,
        H3m3,
        H3p3);

    MPI_Finalize();
    return 0;
}
#else
TEST(Communication, comm_2_test_halo_exchange_3D_generic_full) {
    bool passed = true;

    //const int Nx = 98*2;
    //const int Ny = 54*3;
    //const int Nz = 87*2;
    const int Nx = 260;
    const int Ny = 260;
    const int Nz = 80;

#ifdef __CUDACC__
    gridtools::ghex::tl::mpi::communicator_base mpi_comm;
    int num_devices_per_node;
    cudaGetDeviceCount(&num_devices_per_node);
    MPI_Comm raw_local_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_comm.rank(), MPI_INFO_NULL, &raw_local_comm);
    gridtools::ghex::tl::mpi::communicator_base local_comm(raw_local_comm, gridtools::ghex::tl::mpi::comm_take_ownership);
    if (local_comm.rank()<num_devices_per_node)
    {
        std::cout << "I am rank " << mpi_comm.rank() << " and I own GPU " 
        << (mpi_comm.rank()/local_comm.size())*num_devices_per_node + local_comm.rank() << std::endl;
        GT_CUDA_CHECK(cudaSetDevice(local_comm.rank()));
#ifndef GHEX_1_PATTERN_BENCHMARK
        passed = halo_exchange_3D_generic_full::test(true, Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
#else
        //passed = halo_exchange_3D_generic_full::test(true, Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1);
        passed = halo_exchange_3D_generic_full::test(true, Nx, Ny, Nz, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0);
#endif
    }
    else
    {
#endif

#ifndef GHEX_1_PATTERN_BENCHMARK
    passed = halo_exchange_3D_generic_full::test(false, Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
#else
    //passed = halo_exchange_3D_generic_full::test(false, Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1);
    passed = halo_exchange_3D_generic_full::test(false, Nx, Ny, Nz, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0);
#endif
#ifdef __CUDACC__
    }
#endif

    EXPECT_TRUE(passed);
}
#endif
