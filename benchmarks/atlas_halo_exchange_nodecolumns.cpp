/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

#if defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST)
#include <gridtools/storage/cpu_kfirst.hpp>
#elif defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST)
#include <gridtools/storage/cpu_ifirst.hpp>
#endif
#ifdef GHEX_CUDACC
#include <gridtools/storage/gpu.hpp>
#endif

#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/meshgenerator.h>
#include <atlas/functionspace.h>
#include <atlas/field.h>
#include <atlas/array.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/glue/atlas/field.hpp>
#include <ghex/glue/atlas/atlas_user_concepts.hpp>
#include <ghex/arch_list.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/common/timer.hpp>

#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <gridtools/common/cuda_util.hpp>
#include <ghex/common/cuda_runtime.hpp>
#endif

#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
#else
using transport = gridtools::ghex::tl::ucx_tag;
#endif
using context_type = gridtools::ghex::tl::context<transport>;


TEST(atlas_integration, halo_exchange_nodecolumns) {

    using timer_type = gridtools::ghex::timer;
    using domain_id_t = int;
    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = gridtools::ghex::unstructured::grid;
#if defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_kfirst;
#elif defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_ifirst;
#endif
    using function_space_t = atlas::functionspace::NodeColumns;
    using cpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, int, storage_traits_cpu, function_space_t>;

    const int n_iter = 50;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    // Output file
    std::stringstream ss_file;
    ss_file << rank;
    std::string filename = "atlas_halo_exchange_nodecolumns_times_" + ss_file.str() + ".txt";
    std::ofstream file(filename.c_str());
    file << "Atlas halo exchange nodecolumns - Timings\n";

    // Timers
    timer_type t_atlas_cpu_local, t_atlas_cpu_global; // Atlas on CPU
    timer_type t_ghex_cpu_local, t_ghex_cpu_global; // GHEX on CPU
    timer_type t_ghex_gpu_local, t_ghex_gpu_global; // GHEX on GPU

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O1280");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 100;

    // Generate functionspace associated to the mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(2));

    // Instantiate domain descriptor
    std::vector<domain_descriptor_t> local_domains{};
    domain_descriptor_t d{rank,
                          mesh.nodes().partition(),
                          mesh.nodes().remote_index(),
                          nb_levels};
    local_domains.push_back(d);

    // Instantiate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{};

    // Instantiate recv domain ids generator
    gridtools::ghex::atlas_recv_domain_ids_gen<int> rdig{};

    // Make patterns
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

    // Make communication object
    auto co = gridtools::ghex::make_communication_object<decltype(patterns)>(context.get_communicator());

    // Fields creation and initialization
    ::atlas::FieldSet atlas_fields;
    atlas_fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_1")));
    atlas_fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_2")));
    atlas_fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_3")));
    atlas_fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_4")));
    auto GHEX_field_1 = gridtools::ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_2 = gridtools::ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_3 = gridtools::ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_4 = gridtools::ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes, 1); // 1 component / scalar field
    {
        auto atlas_field_1_data = atlas::array::make_view<int, 2>(atlas_fields["atlas_field_1"]);
        auto atlas_field_2_data = atlas::array::make_view<int, 2>(atlas_fields["atlas_field_2"]);
        auto atlas_field_3_data = atlas::array::make_view<int, 2>(atlas_fields["atlas_field_3"]);
        auto atlas_field_4_data = atlas::array::make_view<int, 2>(atlas_fields["atlas_field_4"]);
        auto GHEX_field_1_data = GHEX_field_1.host_view();
        auto GHEX_field_2_data = GHEX_field_2.host_view();
        auto GHEX_field_3_data = GHEX_field_3.host_view();
        auto GHEX_field_4_data = GHEX_field_4.host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
            for (auto level = 0; level < fs_nodes.levels(); ++level) {
                auto value = (rank << 15) + (node << 7) + level;
                atlas_field_1_data(node, level) = value;
                atlas_field_2_data(node, level) = value;
                atlas_field_3_data(node, level) = value;
                atlas_field_4_data(node, level) = value;
                GHEX_field_1_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_2_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_3_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_4_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
            }
        }
    }

    // GHEX target views
    auto GHEX_field_1_target_data = GHEX_field_1.target_view();
    auto GHEX_field_2_target_data = GHEX_field_2.target_view();
    auto GHEX_field_3_target_data = GHEX_field_3.target_view();
    auto GHEX_field_4_target_data = GHEX_field_4.target_view();

    // Instantiate data descriptor
    cpu_data_descriptor_t data_1{local_domains.front(), GHEX_field_1_target_data, GHEX_field_1.components()};
    cpu_data_descriptor_t data_2{local_domains.front(), GHEX_field_2_target_data, GHEX_field_2.components()};
    cpu_data_descriptor_t data_3{local_domains.front(), GHEX_field_3_target_data, GHEX_field_3.components()};
    cpu_data_descriptor_t data_4{local_domains.front(), GHEX_field_4_target_data, GHEX_field_4.components()};

    // Atlas halo exchange
    // Atlas built-in halo exchange function is called (only from the CPU) for testing data correctness.
    // Time comparison might give a hint that GHEX exchange times are consistent,
    // but Atlas times should not be considered as a baseline.
    fs_nodes.haloExchange(atlas_fields); // first iteration
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        fs_nodes.haloExchange(atlas_fields);
        t_local.toc();
        t_atlas_cpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_atlas_cpu_global(t_global);
    }

    // GHEX halo exchange
    auto h = co.exchange(patterns(data_1), patterns(data_2), patterns(data_3), patterns(data_4)); // first iteration
    h.wait();
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h = co.exchange(patterns(data_1), patterns(data_2), patterns(data_3), patterns(data_4));
        h.wait();
        t_local.toc();
        t_ghex_cpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ghex_cpu_global(t_global);
    }

    // test for correctness
    {
        auto atlas_field_1_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_1"]);
        auto atlas_field_2_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_2"]);
        auto atlas_field_3_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_3"]);
        auto atlas_field_4_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_4"]);
        auto GHEX_field_1_data = GHEX_field_1.const_host_view();
        auto GHEX_field_2_data = GHEX_field_2.const_host_view();
        auto GHEX_field_3_data = GHEX_field_3.const_host_view();
        auto GHEX_field_4_data = GHEX_field_4.const_host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
            for (auto level = 0; level < fs_nodes.levels(); ++level) {
                EXPECT_TRUE(GHEX_field_1_data(node, level, 0) == atlas_field_1_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_2_data(node, level, 0) == atlas_field_2_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_3_data(node, level, 0) == atlas_field_3_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_4_data(node, level, 0) == atlas_field_4_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
            }
        }
    }

    // Write timings
    file << "- Atlas CPU benchmark\n"
        << "\tlocal time = " << t_atlas_cpu_local.mean() / 1000.0 << "+/-" << t_atlas_cpu_local.stddev() / (sqrt(t_atlas_cpu_local.num_samples()) * 1000.0) << "s\n"
        << "\tglobal time = " << t_atlas_cpu_global.mean() / 1000.0 << "+/-" << t_atlas_cpu_global.stddev() / (sqrt(t_atlas_cpu_global.num_samples()) * 1000.0) << "s\n";

    file << "- GHEX CPU benchmark\n"
        << "\tlocal time = " << t_ghex_cpu_local.mean() / 1000.0 << "+/-" << t_ghex_cpu_local.stddev() / (sqrt(t_ghex_cpu_local.num_samples()) * 1000.0) << "s\n"
        << "\tglobal time = " << t_ghex_cpu_global.mean() / 1000.0 << "+/-" << t_ghex_cpu_global.stddev() / (sqrt(t_ghex_cpu_global.num_samples()) * 1000.0) << "s\n";

#ifdef GHEX_CUDACC

    using storage_traits_gpu = gridtools::storage::gpu;

    // Additional data descriptor type for GPU
    using gpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::gpu, domain_id_t, int, storage_traits_gpu, function_space_t>;

    // Additional fields for GPU halo exchange
    auto GHEX_field_1_gpu = gridtools::ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_2_gpu = gridtools::ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_3_gpu = gridtools::ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes, 1); // 1 component / scalar field
    auto GHEX_field_4_gpu = gridtools::ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes, 1); // 1 component / scalar field
    {
        auto GHEX_field_1_gpu_data = GHEX_field_1_gpu.host_view();
        auto GHEX_field_2_gpu_data = GHEX_field_2_gpu.host_view();
        auto GHEX_field_3_gpu_data = GHEX_field_3_gpu.host_view();
        auto GHEX_field_4_gpu_data = GHEX_field_4_gpu.host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
            for (auto level = 0; level < fs_nodes.levels(); ++level) {
                auto value = (rank << 15) + (node << 7) + level;
                GHEX_field_1_gpu_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_2_gpu_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_3_gpu_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
                GHEX_field_4_gpu_data(node, level, 0) = value; // TO DO: hard-coded 3d view. Should be more flexible
            }
        }
    }

    // GHEX target views
    auto GHEX_field_1_gpu_target_data = GHEX_field_1_gpu.target_view();
    auto GHEX_field_2_gpu_target_data = GHEX_field_2_gpu.target_view();
    auto GHEX_field_3_gpu_target_data = GHEX_field_3_gpu.target_view();
    auto GHEX_field_4_gpu_target_data = GHEX_field_4_gpu.target_view();

    // Additional data descriptor for GPU halo exchange
    gpu_data_descriptor_t data_1_gpu{local_domains.front(), 0, GHEX_field_1_gpu_target_data, GHEX_field_1_gpu.components()};
    gpu_data_descriptor_t data_2_gpu{local_domains.front(), 0, GHEX_field_2_gpu_target_data, GHEX_field_2_gpu.components()};
    gpu_data_descriptor_t data_3_gpu{local_domains.front(), 0, GHEX_field_3_gpu_target_data, GHEX_field_3_gpu.components()};
    gpu_data_descriptor_t data_4_gpu{local_domains.front(), 0, GHEX_field_4_gpu_target_data, GHEX_field_4_gpu.components()};

    // GHEX halo exchange on GPU
    auto h_gpu = co.exchange(patterns(data_1_gpu), patterns(data_2_gpu), patterns(data_3_gpu), patterns(data_4_gpu)); // first iteration
    h_gpu.wait();
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_gpu = co.exchange(patterns(data_1_gpu), patterns(data_2_gpu), patterns(data_3_gpu), patterns(data_4_gpu));
        h_gpu.wait();
        t_local.toc();
        t_ghex_gpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ghex_gpu_global(t_global);
    }

    // Test for correctness
    {
        auto atlas_field_1_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_1"]);
        auto atlas_field_2_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_2"]);
        auto atlas_field_3_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_3"]);
        auto atlas_field_4_data = atlas::array::make_view<const int, 2>(atlas_fields["atlas_field_4"]);
        auto GHEX_field_1_gpu_data = GHEX_field_1_gpu.const_host_view();
        auto GHEX_field_2_gpu_data = GHEX_field_2_gpu.const_host_view();
        auto GHEX_field_3_gpu_data = GHEX_field_3_gpu.const_host_view();
        auto GHEX_field_4_gpu_data = GHEX_field_4_gpu.const_host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
            for (auto level = 0; level < fs_nodes.levels(); ++level) {
                EXPECT_TRUE(GHEX_field_1_gpu_data(node, level, 0) == atlas_field_1_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_2_gpu_data(node, level, 0) == atlas_field_2_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_3_gpu_data(node, level, 0) == atlas_field_3_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
                EXPECT_TRUE(GHEX_field_4_gpu_data(node, level, 0) == atlas_field_4_data(node, level)); // TO DO: hard-coded 3d view. Should be more flexible
            }
        }
    }

    // Write timings
    file << "- GHEX GPU benchmark\n"
        << "\tlocal time = " << t_ghex_gpu_local.mean() / 1000.0 << "+/-" << t_ghex_gpu_local.stddev() / (sqrt(t_ghex_gpu_local.num_samples()) * 1000.0) << "s\n"
        << "\tglobal time = " << t_ghex_gpu_global.mean() / 1000.0 << "+/-" << t_ghex_gpu_global.stddev() / (sqrt(t_ghex_gpu_global.num_samples()) * 1000.0) << "s\n";

#endif

}
