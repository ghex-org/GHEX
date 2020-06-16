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
#include <fstream>
#include <vector>

#include <gtest/gtest.h>

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
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/glue/atlas/atlas_user_concepts.hpp>
#include <ghex/arch_list.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/common/timer.hpp>

#ifdef __CUDACC__
#include <gridtools/common/cuda_util.hpp>
#endif

#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif
using context_type = gridtools::ghex::tl::context<transport, threading>;


/* WARN: Atlas halo exchange on GPU is disabled for the moment
 * due to some fixes that has been done in Atlas develop branch
 * and still need to be tested properly here*/
TEST(atlas_integration, halo_exchange_nodecolumns) {

    using timer_type = gridtools::ghex::timer;
    using domain_id_t = int;
    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<domain_id_t>;
    using cpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, int>;

    const int n_iter = 50;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
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
    // timer_type t_atlas_gpu_local, t_atlas_gpu_global; // Atlas on GPU
    timer_type t_ghex_gpu_local, t_ghex_gpu_global; // GHEX on GPU

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O1280");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 100;

    // Generate functionspace associated to the mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

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
    using grid_type = gridtools::ghex::unstructured::grid;
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

    // Make communication object
    auto co = gridtools::ghex::make_communication_object<decltype(patterns)>(context.get_communicator(context.get_token()));

    // Fields creation and initialization
    atlas::FieldSet fields;
    fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_1")));
    fields.add(fs_nodes.createField<int>(atlas::option::name("GHEX_field_1")));
    auto atlas_field_1_data = atlas::array::make_view<int, 2>(fields["atlas_field_1"]);
    auto GHEX_field_1_data = atlas::array::make_view<int, 2>(fields["GHEX_field_1"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            atlas_field_1_data(node, level) = value;
            GHEX_field_1_data(node, level) = value;
        }
    }

    // Instantiate data descriptor
    cpu_data_descriptor_t data_1{local_domains.front(), fields["GHEX_field_1"]};

    // Atlas halo exchange
    fs_nodes.haloExchange(fields["atlas_field_1"]); // first iteration
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        fs_nodes.haloExchange(fields["atlas_field_1"]);
        t_local.toc();
        t_atlas_cpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_atlas_cpu_global(t_global);
    }

    // GHEX halo exchange
    auto h = co.exchange(patterns(data_1)); // first iteration
    h.wait();
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h = co.exchange(patterns(data_1));
        h.wait();
        t_local.toc();
        t_ghex_cpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ghex_cpu_global(t_global);
    }

    // test for correctness
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
        }
    }

    // Write timings
    file << "- Atlas CPU benchmark\n"
        << "\tlocal time = " << t_atlas_cpu_local.mean() / 1000.0 << "+/-" << t_atlas_cpu_local.stddev() / 1000.0 << "s\n"
        << "\tglobal time = " << t_atlas_cpu_global.mean() / 1000.0 << "+/-" << t_atlas_cpu_global.stddev() / 1000.0 << "s\n";

    file << "- GHEX CPU benchmark\n"
        << "\tlocal time = " << t_ghex_cpu_local.mean() / 1000.0 << "+/-" << t_ghex_cpu_local.stddev() / 1000.0 << "s\n"
        << "\tglobal time = " << t_ghex_cpu_global.mean() / 1000.0 << "+/-" << t_ghex_cpu_global.stddev() / 1000.0 << "s\n";

#ifdef __CUDACC__

    // Additional data descriptor type for GPU
    using gpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::gpu, domain_id_t, int>;

    // Additional fields for GPU halo exchange
    // fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_1_gpu")));
    fields.add(fs_nodes.createField<int>(atlas::option::name("GHEX_field_1_gpu")));
    // auto atlas_field_1_gpu_data = atlas::array::make_host_view<int, 2>(fields["atlas_field_1_gpu"]);
    auto GHEX_field_1_gpu_data = atlas::array::make_host_view<int, 2>(fields["GHEX_field_1_gpu"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            // atlas_field_1_gpu_data(node, level) = value;
            GHEX_field_1_gpu_data(node, level) = value;
        }
    }
    // fields["atlas_field_1_gpu"].cloneToDevice();
    fields["GHEX_field_1_gpu"].cloneToDevice();

    // Additional data descriptor for GPU halo exchange
    gpu_data_descriptor_t data_1_gpu{local_domains.front(), 0, fields["GHEX_field_1_gpu"]};

    // Atlas halo exchange on GPU
    // fs_nodes.haloExchange(fields["atlas_field_1_gpu"], true); // first iteration
    // for (auto i = 0; i < n_iter; ++i) { // benchmark
    //     timer_type t_local;
    //     MPI_Barrier(context.mpi_comm());
    //     t_local.tic();
    //     fs_nodes.haloExchange(fields["atlas_field_1_gpu"], true);
    //     t_local.toc();
    //     t_atlas_gpu_local(t_local);
    //     MPI_Barrier(context.mpi_comm());
    //     auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
    //     t_atlas_gpu_global(t_global);
    // }

    // GHEX halo exchange on GPU
    auto h_gpu = co.exchange(patterns(data_1_gpu)); // first iteration
    h_gpu.wait();
    for (auto i = 0; i < n_iter; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_gpu = co.exchange(patterns(data_1_gpu));
        h_gpu.wait();
        t_local.toc();
        t_ghex_gpu_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ghex_gpu_global(t_global);
    }

    // Test for correctness
    // fields["atlas_field_1_gpu"].cloneFromDevice();
    fields["GHEX_field_1_gpu"].cloneFromDevice();
    // fields["atlas_field_1_gpu"].reactivateHostWriteViews();
    fields["GHEX_field_1_gpu"].reactivateHostWriteViews();
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_data(node, level));
            // Not strictly needed, just double check Atlas on GPU
            // EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_gpu_data(node, level));
        }
    }

    // Write timings
    // file << "- Atlas GPU benchmark\n"
    //     << "\tlocal time = " << t_atlas_gpu_local.mean() / 1000.0 << "+/-" << t_atlas_gpu_local.stddev() / 1000.0 << "s\n"
    //     << "\tglobal time = " << t_atlas_gpu_global.mean() / 1000.0 << "+/-" << t_atlas_gpu_global.stddev() / 1000.0 << "s\n";

    file << "- GHEX GPU benchmark\n"
        << "\tlocal time = " << t_ghex_gpu_local.mean() / 1000.0 << "+/-" << t_ghex_gpu_local.stddev() / 1000.0 << "s\n"
        << "\tglobal time = " << t_ghex_gpu_global.mean() / 1000.0 << "+/-" << t_ghex_gpu_global.stddev() / 1000.0 << "s\n";
#endif

}
