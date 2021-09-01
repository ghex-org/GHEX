/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

//#define GHEX_1_PATTERN_BENCHMARK

#ifndef STANDALONE
#include "gtest/gtest.h"
#endif
#include <fstream>
#include <gridtools/common/boollist.hpp>
#include <gridtools/gcl/halo_exchange.hpp>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <sys/time.h>
#include <iomanip>

#include "../utils/triplet.hpp"
#include <ghex/transport_layer/mpi/communicator_base.hpp>
#include <ghex/common/timer.hpp>
//#include "triplet.hpp"

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <gridtools/common/cuda_util.hpp>
#include <ghex/common/cuda_runtime.hpp>
#endif

namespace halo_exchange_3D_generic_full {
    int pid;
    int nprocs;
    MPI_Comm CartComm;
    int dims[3] = {0, 0, 0};
    int coords[3] = {0, 0, 0};

    using timer_type = gridtools::ghex::timer;

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

#ifdef GHEX_CUDACC
    typedef gridtools::gcl::gpu arch_type;
#else
    typedef gridtools::gcl::cpu arch_type;
#endif

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
        triple_t<USE_DOUBLE, T3> *_c) {

        typedef gridtools::layout_map<I1, I2, I3> layoutmap;

        gridtools::ghex::tl::mpi::communicator_base world;

        array<triple_t<USE_DOUBLE, T1>, layoutmap> a(
            _a, (DIM1 + H1m1 + H1p1), (DIM2 + H2m1 + H2p1), (DIM3 + H3m1 + H3p1));
        array<triple_t<USE_DOUBLE, T2>, layoutmap> b(
            _b, (DIM1 + H1m2 + H1p2), (DIM2 + H2m2 + H2p2), (DIM3 + H3m2 + H3p2));
        array<triple_t<USE_DOUBLE, T3>, layoutmap> c(
            _c, (DIM1 + H1m3 + H1p3), (DIM2 + H2m3 + H2p3), (DIM3 + H3m3 + H3p3));

        /* The pattern type is defined with the layouts, data types and
           number of dimensions.

           The logical assumption done in the program is that 'i' is the
           first dimension (rows), 'j' is the second, and 'k' is the
           third. The first layout states that 'i' is the second dimension
           in order of strides, while 'j' is the first and 'k' is the third
           (just by looking at the initialization loops this shoule be
           clear).

           The second layout states that the first dimension in data ('i')
           identify also the first dimension in the communicator. Logically,
           moving on 'i' dimension from processot (p,q,r) will lead you
           logically to processor (p+1,q,r). The other dimensions goes as
           the others.
        */
        typedef gridtools::gcl::halo_exchange_generic<gridtools::layout_map<0, 1, 2>, arch_type> pattern_type;

        /* The pattern is now instantiated with the periodicities and the
           communicator. The periodicity of the communicator is
           irrelevant. Setting it to be periodic is the best choice, then
           GCL can deal with any periodicity easily.
        */
        pattern_type he(typename pattern_type::grid_type::period_type(per0, per1, per2), CartComm);

        gridtools::array<gridtools::halo_descriptor, 3> halo_dsc1;
        halo_dsc1[0] = gridtools::halo_descriptor(H1m1, H1p1, H1m1, DIM1 + H1m1 - 1, DIM1 + H1m1 + H1p1);
        halo_dsc1[1] = gridtools::halo_descriptor(H2m1, H2p1, H2m1, DIM2 + H2m1 - 1, DIM2 + H2m1 + H2p1);
        halo_dsc1[2] = gridtools::halo_descriptor(H3m1, H3p1, H3m1, DIM3 + H3m1 - 1, DIM3 + H3m1 + H3p1);

        gridtools::array<gridtools::halo_descriptor, 3> halo_dsc2;
        halo_dsc2[0] = gridtools::halo_descriptor(H1m2, H1p2, H1m2, DIM1 + H1m2 - 1, DIM1 + H1m2 + H1p2);
        halo_dsc2[1] = gridtools::halo_descriptor(H2m2, H2p2, H2m2, DIM2 + H2m2 - 1, DIM2 + H2m2 + H2p2);
        halo_dsc2[2] = gridtools::halo_descriptor(H3m2, H3p2, H3m2, DIM3 + H3m2 - 1, DIM3 + H3m2 + H3p2);

        gridtools::array<gridtools::halo_descriptor, 3> halo_dsc3;
        halo_dsc3[0] = gridtools::halo_descriptor(H1m3, H1p3, H1m3, DIM1 + H1m3 - 1, DIM1 + H1m3 + H1p3);
        halo_dsc3[1] = gridtools::halo_descriptor(H2m3, H2p3, H2m3, DIM2 + H2m3 - 1, DIM2 + H2m3 + H2p3);
        halo_dsc3[2] = gridtools::halo_descriptor(H3m3, H3p3, H3m3, DIM3 + H3m3 - 1, DIM3 + H3m3 + H3p3);

        /* Pattern is set up. This must be done only once per pattern. The
           parameter must me greater or equal to the largest number of
           arrays updated in a single step.
        */
        // he.setup(100, halo_dsc, sizeof(double));

        gridtools::array<gridtools::halo_descriptor, 3> h_example;
#define MAX3(a, b, c) std::max(a, std::max(b, c))
        h_example[0] = gridtools::halo_descriptor(MAX3(H1m1, H1m2, H1m3),
            MAX3(H1p1, H1p2, H1p3),
            MAX3(H1m1, H1m2, H1m3),
            DIM1 + MAX3(H1m1, H1m2, H1m3) - 1,
            DIM1 + MAX3(H1m1, H1m2, H1m3) + MAX3(H1p1, H1p3, H1p3));
        h_example[1] = gridtools::halo_descriptor(MAX3(H2m1, H2m2, H2m3),
            MAX3(H2p1, H2p2, H2p3),
            MAX3(H2m1, H2m2, H2m3),
            DIM2 + MAX3(H2m1, H2m2, H2m3) - 1,
            DIM2 + MAX3(H2m1, H2m2, H2m3) + MAX3(H2p1, H2p3, H2p3));
        h_example[2] = gridtools::halo_descriptor(MAX3(H3m1, H3m2, H3m3),
            MAX3(H3p1, H3p2, H3p3),
            MAX3(H3m1, H3m2, H3m3),
            DIM3 + MAX3(H3m1, H3m2, H3m3) - 1,
            DIM3 + MAX3(H3m1, H3m2, H3m3) + MAX3(H3p1, H3p3, H3p3));
#undef MAX3
        he.setup(3,
            gridtools::gcl::field_on_the_fly<int, layoutmap, pattern_type::traits>(nullptr, h_example), // BEWARE!!!!
            std::max(sizeof(triple_t<USE_DOUBLE, T1>::data_type),
                std::max(sizeof(triple_t<USE_DOUBLE, T2>::data_type),
                    sizeof(triple_t<USE_DOUBLE, T3>::data_type)) // Estimates the size
                ));

        file << "Proc: (" << coords[0] << ", " << coords[1] << ", " << coords[2] << ")\n";

        /* Just an initialization */
        for (int ii = 0; ii < DIM1 + H1m1 + H1p1; ++ii)
            for (int jj = 0; jj < DIM2 + H2m1 + H2p1; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m1 + H3p1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m2 + H1p2; ++ii)
            for (int jj = 0; jj < DIM2 + H2m2 + H2p2; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m2 + H3p2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>();
                }
            }

        for (int ii = 0; ii < DIM1 + H1m3 + H1p3; ++ii)
            for (int jj = 0; jj < DIM2 + H2m3 + H2p3; ++jj) {
                for (int kk = 0; kk < DIM3 + H3m3 + H3p3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>();
                }
            }

        for (int ii = H1m1; ii < DIM1 + H1m1; ++ii)
            for (int jj = H2m1; jj < DIM2 + H2m1; ++jj)
                for (int kk = H3m1; kk < DIM3 + H3m1; ++kk) {
                    a(ii, jj, kk) = triple_t<USE_DOUBLE, T1>(
                        ii - H1m1 + (DIM1)*coords[0], jj - H2m1 + (DIM2)*coords[1], kk - H3m1 + (DIM3)*coords[2]);
                }

        for (int ii = H1m2; ii < DIM1 + H1m2; ++ii)
            for (int jj = H2m2; jj < DIM2 + H2m2; ++jj)
                for (int kk = H3m2; kk < DIM3 + H3m2; ++kk) {
                    b(ii, jj, kk) = triple_t<USE_DOUBLE, T2>(ii - H1m2 + (DIM1)*coords[0] + B_ADD,
                        jj - H2m2 + (DIM2)*coords[1] + B_ADD,
                        kk - H3m2 + (DIM3)*coords[2] + B_ADD);
                }

        for (int ii = H1m3; ii < DIM1 + H1m3; ++ii)
            for (int jj = H2m3; jj < DIM2 + H2m3; ++jj)
                for (int kk = H3m3; kk < DIM3 + H3m3; ++kk) {
                    c(ii, jj, kk) = triple_t<USE_DOUBLE, T3>(ii - H1m3 + (DIM1)*coords[0] + C_ADD,
                        jj - H2m3 + (DIM2)*coords[1] + C_ADD,
                        kk - H3m3 + (DIM3)*coords[2] + C_ADD);
                }

        file << "A \n";
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();

#ifdef GHEX_CUDACC
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
            a.ptr,
            (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) *
                sizeof(triple_t<USE_DOUBLE, T1>::data_type),
            cudaMemcpyHostToDevice));

        GT_CUDA_CHECK(cudaMemcpy(gpu_b,
            b.ptr,
            (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) *
                sizeof(triple_t<USE_DOUBLE, T2>::data_type),
            cudaMemcpyHostToDevice));

        GT_CUDA_CHECK(cudaMemcpy(gpu_c,
            c.ptr,
            (DIM1 + H1m3 + H1p3) * (DIM2 + H2m3 + H2p3) * (DIM3 + H3m3 + H3p3) *
                sizeof(triple_t<USE_DOUBLE, T3>::data_type),
            cudaMemcpyHostToDevice));

        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T1>::data_type, layoutmap, pattern_type::traits> field1(
            reinterpret_cast<triple_t<USE_DOUBLE, T1>::data_type *>(gpu_a), halo_dsc1);
        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T2>::data_type, layoutmap, pattern_type::traits> field2(
            reinterpret_cast<triple_t<USE_DOUBLE, T2>::data_type *>(gpu_b), halo_dsc2);
        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T3>::data_type, layoutmap, pattern_type::traits> field3(
            reinterpret_cast<triple_t<USE_DOUBLE, T3>::data_type *>(gpu_c), halo_dsc3);
#else
        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T1>::data_type, layoutmap, pattern_type::traits> field1(
            reinterpret_cast<triple_t<USE_DOUBLE, T1>::data_type *>(a.ptr), halo_dsc1);
        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T2>::data_type, layoutmap, pattern_type::traits> field2(
            reinterpret_cast<triple_t<USE_DOUBLE, T2>::data_type *>(b.ptr), halo_dsc2);
        gridtools::gcl::field_on_the_fly<triple_t<USE_DOUBLE, T3>::data_type, layoutmap, pattern_type::traits> field3(
            reinterpret_cast<triple_t<USE_DOUBLE, T3>::data_type *>(c.ptr), halo_dsc3);
#endif


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
#ifdef VECTOR_INTERFACE
            world.barrier();
            t_0.tic();
            he.pack(vect);
            t_0.toc();
            t_1.tic();
            he.exchange();
            he.unpack(vect);
            t_1.toc();
            world.barrier();
#else
            world.barrier();
            t_0.tic();
            he.pack(field1, field2, field3);
            t_0.toc();
            t_1.tic();
            he.exchange();
            he.unpack(field1, field2, field3);
            t_1.toc();
            world.barrier();
#endif
            timer_type t;
            t(t_0.sum()+t_1.sum());

            auto t_0_all = gridtools::ghex::reduce(t_0,world);
            auto t_1_all = gridtools::ghex::reduce(t_1,world);
            auto t_all = gridtools::ghex::reduce(t,world);
            if (k >= k_start)
            {
                t_0_local(t_0);
                t_1_local(t_1);
                t_local(t);
                t_0_global(t_0_all);
                t_1_global(t_1_all);
                t_global(t_all);
            }

            file << "TIME PACK:        " 
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
        file << "TIME PACK:        " 
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

#ifdef GHEX_CUDACC
        GT_CUDA_CHECK(cudaMemcpy(a.ptr,
            gpu_a,
            (DIM1 + H1m1 + H1p1) * (DIM2 + H2m1 + H2p1) * (DIM3 + H3m1 + H3p1) *
                sizeof(triple_t<USE_DOUBLE, T1>::data_type),
            cudaMemcpyDeviceToHost));

        GT_CUDA_CHECK(cudaMemcpy(b.ptr,
            gpu_b,
            (DIM1 + H1m2 + H1p2) * (DIM2 + H2m2 + H2p2) * (DIM3 + H3m2 + H3p2) *
                sizeof(triple_t<USE_DOUBLE, T2>::data_type),
            cudaMemcpyDeviceToHost));

        GT_CUDA_CHECK(cudaMemcpy(c.ptr,
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
        printbuff(file, a, DIM1 + H1m1 + H1p1, DIM2 + H2m1 + H2p1, DIM3 + H3m1 + H3p1);
        file << "B \n";
        printbuff(file, b, DIM1 + H1m2 + H1p2, DIM2 + H2m2 + H2p2, DIM3 + H3m2 + H3p2);
        file << "C \n";
        printbuff(file, c, DIM1 + H1m3 + H1p3, DIM2 + H2m3 + H2p3, DIM3 + H3m3 + H3p3);
        file.flush();

        int passed = true;

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

                    if (a(ii, jj, kk) != ta) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "a " << a(ii, jj, kk) << " != " << ta << "\n";
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

                    if (b(ii, jj, kk) != tb) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "b " << b(ii, jj, kk) << " != " << tb << "\n";
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

                    if (c(ii, jj, kk) != tc) {
                        passed = false;
                        file << ii << ", " << jj << ", " << kk << " values found != expected: "
                             << "c " << c(ii, jj, kk) << " != " << tc << "\n";
                    }
                }

        if (passed)
            file << "RESULT: PASSED!\n";
        else
            file << "RESULT: FAILED!\n";

        return passed;
    }

    bool test(int DIM1,
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
        int H3p3) {

        /* Here we compute the computing gris as in many applications
         */
        MPI_Comm_rank(MPI_COMM_WORLD, &pid);
        MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
        //std::cout << pid << " " << nprocs << "\n";

        std::stringstream ss;
        ss << pid;
        std::string filename = "gcl_out" + ss.str() + ".txt";
        //std::cout << filename << std::endl;
        std::ofstream file(filename.c_str());

        file << pid << "  " << nprocs << "\n";

        dims[2]=1;
        MPI_Dims_create(nprocs, 3, dims);
        int period[3] = {1, 1, 1};

        file << "@" << pid << "@ MPI GRID SIZE " << dims[0] << " - " << dims[1] << " - " << dims[2] << "\n";

        MPI_Cart_create(MPI_COMM_WORLD, 3, dims, period, false, &CartComm);

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
int main(int argc, char **argv) {
#ifdef GT_USE_GPU
    device_binding();
#endif

    MPI_Init(&argc, &argv);
    gridtools::GCL_Init(argc, argv);

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
}
#else
TEST(Communication, gcl_test_halo_exchange_3D_generic_full) {
    //const int Nx = 98*2;
    //const int Ny = 54*3;
    //const int Nz = 87*2;
    const int Nx = 260;
    const int Ny = 260;
    const int Nz = 80;
#ifndef GHEX_1_PATTERN_BENCHMARK
    bool passed = halo_exchange_3D_generic_full::test(Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 0, 1);
#else
    //bool passed = halo_exchange_3D_generic_full::test(Nx, Ny, Nz, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1);
    bool passed = halo_exchange_3D_generic_full::test(Nx, Ny, Nz, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0, 3, 3, 3, 3, 0, 0);
#endif
    EXPECT_TRUE(passed);
}
#endif
