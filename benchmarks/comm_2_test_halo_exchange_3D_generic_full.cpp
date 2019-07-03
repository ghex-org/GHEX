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

#define GRIDTOOLS_GHEX_TIMINGS


#ifndef STANDALONE
    #include "gtest/gtest.h"
//#define GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI
    #include "gtest_main_boost.cpp"
#endif
#include <boost/mpi/communicator.hpp>
#include <boost/mpi/collectives.hpp>
#include <sstream>
#include <string>
#include <fstream>
#include <iomanip>
#include <chrono>

#include "../include/communication_object_2.hpp"
#include "../include/structured_pattern.hpp"
#include "../include/structured_domain_descriptor.hpp"
#include "../include/simple_field.hpp"
#include <array>

#include "triplet.hpp"
#include <gridtools/tools/mpi_unit_test_driver/device_binding.hpp>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/moment.hpp>
#include <boost/accumulators/statistics/variance.hpp>
#include <boost/accumulators/statistics/max.hpp>
#include <boost/accumulators/statistics/min.hpp>

#ifdef __CUDACC__
#include <gridtools/common/cuda_util.hpp>
#endif

namespace halo_exchange_3D_generic_full {

    using clock_type = std::chrono::high_resolution_clock;
    using duration_type = typename clock_type::duration;
    using time_point_type = typename clock_type::time_point;
    using microseconds = std::chrono::microseconds;

#ifdef GRIDTOOLS_GHEX_TIMINGS
    struct timings_type
    {
        time_point_type t0;
        time_point_type t1;
        time_point_type t2;
        duration_type allocation_part = duration_type(0);
        duration_type malloc_part     = duration_type(0);
        duration_type resize_part     = duration_type(0);
    };
#endif

    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

#define B_ADD 1
#define C_ADD 2

    typedef int T1;
    typedef double T2;
    typedef long long int T3;

    using domain_descriptor_type = gridtools::structured_domain_descriptor<int,3>;
    template<typename T, typename Device, int... Is>
    using field_descriptor_type  = gridtools::simple_field_wrapper<T,Device,domain_descriptor_type, Is...>;

#ifdef __CUDACC__
    using arch_type = gridtools::device::gpu;
#else
    using arch_type = gridtools::device::cpu;
#endif
    
    template<typename T, typename Device, typename DomainDescriptor, int... Order>
    void printbuff(std::ostream& file, const gridtools::simple_field_wrapper<T,Device,DomainDescriptor, Order...>& field)
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

    template <typename ST, int I1, int I2, int I3, bool per0, bool per1, bool per2>
    bool run(ST &file,
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
        triple_t<USE_DOUBLE, T3> *_c) 
    {
        using TT1 = triple_t<USE_DOUBLE, T1>;
        using TT2 = triple_t<USE_DOUBLE, T2>;
        using TT3 = triple_t<USE_DOUBLE, T3>;

        // mpi communicator
        //boost::mpi::communicator comm(CartComm,boost::mpi::comm_attach);
        boost::mpi::communicator world;
        
        // compute total domain
        const std::array<int,3> g_first{             0,              0,              0};
        const std::array<int,3> g_last {dims[0]*DIM1-1, dims[1]*DIM2-1, dims[2]*DIM3-1};

        // periodicity
        const std::array<bool,3> periodic{per0,per1,per2};

        // halos
        const std::array<int,6> halo_1{H1m1,H1p1,H2m1,H2p1,H3m1,H3p1};
        const std::array<int,6> halo_2{H1m2,H1p2,H2m2,H2p2,H3m2,H3p2};
        const std::array<int,6> halo_3{H1m3,H1p3,H2m3,H2p3,H3m3,H3p3};

        // define local domain
        domain_descriptor_type local_domain{
            world.rank(),//comm.rank(),
            std::array<int,3>{coords[0]*DIM1,coords[1]*DIM2,coords[2]*DIM3},
            std::array<int,3>{(coords[0]+1)*DIM1-1,(coords[1]+1)*DIM2-1,(coords[2]+1)*DIM3-1}};
        std::vector<domain_descriptor_type> local_domains{local_domain};
        
        // wrap raw fields
        field_descriptor_type<TT1, gridtools::device::cpu, I1, I2, I3> a(local_domain.domain_id(), _a, 
            std::array<int,3>{H1m1,H2m1,H3m1},
            std::array<int,3>{(DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1)});

        field_descriptor_type<TT2, gridtools::device::cpu, I1, I2, I3> b(local_domain.domain_id(), _b, 
            std::array<int,3>{H1m2,H2m2,H3m2},
            std::array<int,3>{(DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2)});

        field_descriptor_type<TT3, gridtools::device::cpu, I1, I2, I3> c(local_domain.domain_id(), _c, 
            std::array<int,3>{H1m3,H2m3,H3m3},
            std::array<int,3>{(DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3)});

        // make halo generators
        auto halo_gen_1 = domain_descriptor_type::halo_generator_type(g_first, g_last, halo_1, periodic);
        auto halo_gen_2 = domain_descriptor_type::halo_generator_type(g_first, g_last, halo_2, periodic);
        auto halo_gen_3 = domain_descriptor_type::halo_generator_type(g_first, g_last, halo_3, periodic);

        // make patterns
        auto pattern_1 = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen_1, local_domains);
        auto pattern_2 = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen_2, local_domains);
        auto pattern_3 = gridtools::make_pattern<gridtools::structured_grid>(world, halo_gen_3, local_domains);
        
        // communication object
        auto co = gridtools::make_communication_object(pattern_1, pattern_2, pattern_3);


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

#ifdef __CUDACC__
        file << "***** GPU ON *****\n";

        triple_t<USE_DOUBLE, T1>::data_type *gpu_a = 0;
        triple_t<USE_DOUBLE, T2>::data_type *gpu_b = 0;
        triple_t<USE_DOUBLE, T3>::data_type *gpu_c = 0;
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

        // wrap raw fields
        field_descriptor_type<TT1, gridtools::device::gpu, I1, I2, I3> field1(local_domain.domain_id(), gpu_a, 
            std::array<int,3>{H1m1,H2m1,H3m1},
            std::array<int,3>{(DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1)});

        field_descriptor_type<TT2, gridtools::device::gpu, I1, I2, I3> field2(local_domain.domain_id(), gpu_b, 
            std::array<int,3>{H1m2,H2m2,H3m2},
            std::array<int,3>{(DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2)});

        field_descriptor_type<TT3, gridtools::device::gpu, I1, I2, I3> field3(local_domain.domain_id(), gpu_c, 
            std::array<int,3>{H1m3,H2m3,H3m3},
            std::array<int,3>{(DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3)});

#else
        auto field1 = a;
        auto field2 = b;
        auto field3 = c;
#endif

        file << "                         LOCAL        MEAN          STD         MIN         MAX" << std::endl;
        using namespace boost::accumulators;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_0;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_1;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_0;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_1;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global;
#ifdef GRIDTOOLS_GHEX_TIMINGS
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_2;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_3;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_2;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_3;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_4;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_5;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_local_6;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_4;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_5;
        accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > time_acc_global_6;
#endif
        const int k_start = 5;
        for (int k=0; k<20; ++k)
        {
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_0;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_1;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global;
#ifdef GRIDTOOLS_GHEX_TIMINGS
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_2;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_3;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_4;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_5;
            accumulator_set<typename microseconds::rep, stats<tag::mean, tag::variance(lazy), tag::max, tag::min> > acc_global_6;
            timings_type timings;
            timings.allocation_part = duration_type(0);
            timings.malloc_part = duration_type(0);
            timings.resize_part = duration_type(0);
#endif
            world.barrier();
            const auto t0 = clock_type::now();
            auto h = co.exchange(
#ifdef GRIDTOOLS_GHEX_TIMINGS
                timings,
#endif
                pattern_1(field1),
                pattern_2(field2),
                pattern_3(field3));
            const auto t1 = clock_type::now();
            h.wait();
            const auto t2 = clock_type::now();
            world.barrier();

            const auto d0 = std::chrono::duration_cast<microseconds>(t1-t0).count();
            const auto d1 = std::chrono::duration_cast<microseconds>(t2-t1).count();
#ifdef GRIDTOOLS_GHEX_TIMINGS
            const auto d2 = std::chrono::duration_cast<microseconds>(timings.t1-timings.t0).count();
            const auto d3 = std::chrono::duration_cast<microseconds>(timings.t2-timings.t1).count();
            const auto d4 = std::chrono::duration_cast<microseconds>(timings.allocation_part).count();
            const auto d5 = std::chrono::duration_cast<microseconds>(timings.malloc_part).count();
            const auto d6 = std::chrono::duration_cast<microseconds>(timings.resize_part).count();
#endif

            std::vector<typename microseconds::rep> tmp_0;
            boost::mpi::all_gather(world, d0, tmp_0); 
            std::vector<typename microseconds::rep> tmp_1;
            boost::mpi::all_gather(world, d1, tmp_1); 
            std::vector<typename microseconds::rep> tmp;
            boost::mpi::all_gather(world, d0+d1, tmp); 
#ifdef GRIDTOOLS_GHEX_TIMINGS
            std::vector<typename microseconds::rep> tmp_2;
            boost::mpi::all_gather(world, d2, tmp_2); 
            std::vector<typename microseconds::rep> tmp_3;
            boost::mpi::all_gather(world, d3, tmp_3); 
            std::vector<typename microseconds::rep> tmp_4;
            boost::mpi::all_gather(world, d4, tmp_4); 
            std::vector<typename microseconds::rep> tmp_5;
            boost::mpi::all_gather(world, d5, tmp_5); 
            std::vector<typename microseconds::rep> tmp_6;
            boost::mpi::all_gather(world, d6, tmp_6); 
#endif
            for (unsigned int i=0; i<tmp_0.size(); ++i)
            {
                acc_global_0(tmp_0[i]);
                acc_global_1(tmp_1[i]);
                acc_global(tmp[i]);
#ifdef GRIDTOOLS_GHEX_TIMINGS
                acc_global_2(tmp_2[i]);
                acc_global_3(tmp_3[i]);
                acc_global_4(tmp_4[i]);
                acc_global_5(tmp_5[i]);
                acc_global_6(tmp_6[i]);
#endif
                if (k >= k_start)
                {
                    time_acc_global_0(tmp_0[i]);
                    time_acc_global_1(tmp_1[i]);
                    time_acc_global(tmp[i]);
#ifdef GRIDTOOLS_GHEX_TIMINGS
                    time_acc_global_2(tmp_2[i]);
                    time_acc_global_3(tmp_3[i]);
                    time_acc_global_4(tmp_4[i]);
                    time_acc_global_5(tmp_5[i]);
                    time_acc_global_6(tmp_6[i]);
#endif
                }
            }
            if (k >= k_start)
            {
                time_acc_local_0(d0);
                time_acc_local_1(d1);
                time_acc_local(d0+d1);
#ifdef GRIDTOOLS_GHEX_TIMINGS
                time_acc_local_2(d2);
                time_acc_local_3(d3);
                time_acc_local_4(d4);
                time_acc_local_5(d5);
                time_acc_local_6(d6);
#endif
            }

            const auto global_0_mean    = mean(acc_global_0);
            const auto global_1_mean    = mean(acc_global_1);
            const auto global_mean      = mean(acc_global);
            const auto global_0_std_dev = std::sqrt(variance(acc_global_0));
            const auto global_1_std_dev = std::sqrt(variance(acc_global_1));
            const auto global_std_dev   = std::sqrt(variance(acc_global));
            const auto global_0_max     = max(acc_global_0);
            const auto global_1_max     = max(acc_global_1);
            const auto global_max       = max(acc_global);
            const auto global_0_min     = min(acc_global_0);
            const auto global_1_min     = min(acc_global_1);
            const auto global_min       = min(acc_global);

            file << "TIME PACK/POST:   " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d0/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_0_mean/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << global_0_std_dev/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_0_min/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_0_max/1000.0
                << std::endl;
#ifdef GRIDTOOLS_GHEX_TIMINGS
            file << "  TIME SETUP:     " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d2/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(acc_global_2)/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(acc_global_2))/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(acc_global_2)/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(acc_global_2)/1000.0
                << std::endl;
            file << "    TIME ALLOCAT: " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d4/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(acc_global_4)/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(acc_global_4))/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(acc_global_4)/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(acc_global_4)/1000.0
                << std::endl;
            file << "    TIME MALLOC:  " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d5/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(acc_global_5)/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(acc_global_5))/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(acc_global_5)/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(acc_global_5)/1000.0
                << std::endl;
            file << "      TIME RESIZ: " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d6/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(acc_global_6)/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(acc_global_6))/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(acc_global_6)/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(acc_global_6)/1000.0
                << std::endl;
            file << "  TIME PACK:      " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d3/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(acc_global_3)/1000.0
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(acc_global_3))/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(acc_global_3)/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(acc_global_3)/1000.0
                << std::endl;
#endif
            file << "TIME WAIT/UNPACK: " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << d1/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_1_mean/1000.0 
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << global_1_std_dev/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_1_min/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_1_max/1000.0
                << std::endl;
            file << "TIME ALL:         " 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << (d0+d1)/1000.0 
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_mean/1000.0 
                << " ±"
                << std::scientific << std::setprecision(4) << std::right << std::setw(11) << global_std_dev/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_min/1000.0
                << std::scientific << std::setprecision(4) << std::right << std::setw(12) << global_max/1000.0
                << std::endl;
            file << std::endl;
        }

        file << std::endl << "-----------------" << std::endl;
        file << "TIME PACK/POST:   " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_0)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_0)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_0))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_0)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_0)/1000.0
            << std::endl;
#ifdef GRIDTOOLS_GHEX_TIMINGS
        file << "  TIME SETUP:     " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_2)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_2)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_2))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_2)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_2)/1000.0
            << std::endl;
        file << "    TIME ALLOCAT: " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_4)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_4)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_4))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_4)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_4)/1000.0
            << std::endl;
        file << "    TIME MALLOC:  " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_5)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_5)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_5))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_5)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_5)/1000.0
            << std::endl;
        file << "      TIME RESIZ: " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_6)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_6)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_6))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_6)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_6)/1000.0
            << std::endl;
        file << "  TIME PACK:      " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_3)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_3)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_3))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_3)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_3)/1000.0
            << std::endl;
#endif
        file << "TIME WAIT/UNPACK: " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local_1)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global_1)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global_1))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global_1)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global_1)/1000.0
            << std::endl;
        file << "TIME ALL:         " 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_local)/1000.0 
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << mean(time_acc_global)/1000.0
            << " ±"
            << std::scientific << std::setprecision(4) << std::right << std::setw(11) << std::sqrt(variance(time_acc_global))/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << min(time_acc_global)/1000.0
            << std::scientific << std::setprecision(4) << std::right << std::setw(12) << max(time_acc_global)/1000.0
            << std::endl;
        //file << std::endl << std::endl;
        
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
#endif

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
    
    bool test(
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
        boost::mpi::communicator world;
        //std::cout << world.rank() << " " << world.size() << "\n";
        
        std::stringstream ss;
        ss << world.rank();
        std::string filename = "comm_2_out" + ss.str() + ".txt";
        //std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << world.rank() << "  " << world.size() << "\n";
        
        MPI_Dims_create(world.size(), 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << world.rank() << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

        MPI_Cart_create(world, 3, dims, period, false, &CartComm);

        MPI_Cart_get(CartComm, 3, dims, period, coords);

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

        passed = passed and run<std::ostream, 0, 1, 2, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 0,1,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 0,1,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,1,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 0,1,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,1,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,1,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 0,1,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 1, 2, false, false, false>(file,
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
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 0,2,1\n";

        file << "run<std::ostream, 0,2,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 0,2,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 0,2,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,2,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 0,2,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 0,2,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 0,2,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 0, 2, 1, false, false, false>(file,
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
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,0,2\n";

        file << "run<std::ostream, 1,0,2, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 1,0,2, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 1,0,2, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,0,2, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 1,0,2, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,0,2, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 1,0,2, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 0, 2, false, false, false>(file,
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
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 1,2,0\n";

        file << "run<std::ostream, 1,2,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 1,2,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 1,2,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,2,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 1,2,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 1,2,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 1,2,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H31, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 1, 2, 0, false, false, false>(file,
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
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,0,1\n";

        file << "run<std::ostream, 2,0,1, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 2,0,1, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 2,0,1, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,0,1, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 2,0,1, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,0,1, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 2,0,1, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 0, 1, false, false, false>(file,
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
                                _c);
        file << "---------------------------------------------------\n";

        file << "Permutation 2,1,0\n";

        file << "run<std::ostream, 2,1,0, true, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, true>(file,
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
                                _c);

        file << "run<std::ostream, 2,1,0, true, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, true, false>(file,
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
                                _c);

        file << "run<std::ostream, 2,1,0, true, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,1,0, true, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, true, false, false>(file,
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
                                _c);

        file << "run<std::ostream, 2,1,0, false, true, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
                "_b, "
                "_c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, true>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, true, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, true, false>(file,
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
                                _c);

        file
            << "run<std::ostream, 2,1,0, false, false, true>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, _a, "
               "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, true>(file,
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
                                _c);

        file << "run<std::ostream, 2,1,0, false, false, false>(file, DIM1, DIM2, DIM3, H1m, H1p, H2m, H2p, H3m, H3p, "
                "_a, "
                "_b, _c)\n";
        passed = passed and run<std::ostream, 2, 1, 0, false, false, false>(file,
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
                                _c);
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
    boost::mpi::environment env(argc, argv);
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

#ifdef GHEX_BENCHMARKS_USE_MULTI_THREADED_MPI
    MPI_Finalize();
#endif
    return 0;
}
#else
TEST(Communication, comm_2_test_halo_exchange_3D_generic_full) {
    bool passed = halo_exchange_3D_generic_full::test(98, 54, 87, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
    EXPECT_TRUE(passed);
}
#endif

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

