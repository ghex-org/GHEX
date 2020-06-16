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


TEST(atlas_integration, halo_exchange) {

    using domain_id_t = int;
    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = gridtools::ghex::unstructured::grid;
    using cpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, int>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O256");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 10;

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

    // Atlas halo exchange (reference values)
    fs_nodes.haloExchange(fields["atlas_field_1"]);

    // GHEX halo exchange
    auto h = co.exchange(patterns(data_1));
    h.wait();

    // test for correctness
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
        }
    }

#ifdef __CUDACC__
    // Additional data descriptor type for GPU
    using gpu_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::gpu, domain_id_t, int>;

    // Additional field for GPU halo exchange
    fields.add(fs_nodes.createField<int>(atlas::option::name("GHEX_field_1_gpu")));
    auto GHEX_field_1_gpu_data = atlas::array::make_host_view<int, 2>(fields["GHEX_field_1_gpu"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            GHEX_field_1_gpu_data(node, level) = value;
        }
    }
    fields["GHEX_field_1_gpu"].cloneToDevice();

    // Additional data descriptor for GPU halo exchange
    gpu_data_descriptor_t data_1_gpu{local_domains.front(), 0, fields["GHEX_field_1_gpu"]};

    // GHEX halo exchange on GPU
    auto h_gpu = co.exchange(patterns(data_1_gpu));
    h_gpu.wait();

    // Test for correctness
    fields["GHEX_field_1_gpu"].cloneFromDevice();
    fields["GHEX_field_1_gpu"].reactivateHostWriteViews();
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_data(node, level));
        }
    }
#endif

}


TEST(atlas_integration, halo_exchange_multiple_patterns) {

    using domain_id_t = int;
    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = gridtools::ghex::unstructured::grid;
    using cpu_int_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, int>;
    using cpu_double_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, double>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O256");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 10;

    // Generate functionspace associated to the mesh with halo size = 1
    atlas::functionspace::NodeColumns fs_nodes_1(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor (halo size = 1)
    std::vector<domain_descriptor_t> local_domains_1{};
    domain_descriptor_t d_1{rank,
                            mesh.nodes().partition(),
                            mesh.nodes().remote_index(),
                            nb_levels};
    local_domains_1.push_back(d_1);

    // Generate functionspace associated to the mesh with halo size = 2
    atlas::functionspace::NodeColumns fs_nodes_2(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(2));

    // Instantiate domain descriptor (halo size = 1)
    std::vector<domain_descriptor_t> local_domains_2{};
    domain_descriptor_t d_2{rank,
                            mesh.nodes().partition(),
                            mesh.nodes().remote_index(),
                            nb_levels};
    local_domains_2.push_back(d_2);

    // Instantate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{};

    // Instantiate recv domain ids generator
    gridtools::ghex::atlas_recv_domain_ids_gen<int> rdig{};

    // Make patterns
    auto patterns_1 = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains_1);
    auto patterns_2 = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains_2);

    // Make communication object
    auto co = gridtools::ghex::make_communication_object<decltype(patterns_1)>(context.get_communicator(context.get_token()));

    // Fields creation and initialization
    atlas::FieldSet fields_1, fields_2;
    fields_1.add(fs_nodes_1.createField<int>(atlas::option::name("serial_field_1")));
    fields_1.add(fs_nodes_1.createField<int>(atlas::option::name("multi_field_1")));
    fields_2.add(fs_nodes_2.createField<double>(atlas::option::name("serial_field_2")));
    fields_2.add(fs_nodes_2.createField<double>(atlas::option::name("multi_field_2")));
    auto serial_field_1_data = atlas::array::make_view<int, 2>(fields_1["serial_field_1"]);
    auto multi_field_1_data = atlas::array::make_view<int, 2>(fields_1["multi_field_1"]);
    auto serial_field_2_data = atlas::array::make_view<double, 2>(fields_2["serial_field_2"]);
    auto multi_field_2_data = atlas::array::make_view<double, 2>(fields_2["multi_field_2"]);
    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            serial_field_1_data(node, level) = value;
            multi_field_1_data(node, level) = value;
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            auto value = ((rank << 15) + (node << 7) + level) * 0.5;
            serial_field_2_data(node, level) = value;
            multi_field_2_data(node, level) = value;
        }
    }

    // Instantiate data descriptors
    cpu_int_data_descriptor_t serial_data_1{local_domains_1.front(), fields_1["serial_field_1"]};
    cpu_int_data_descriptor_t multi_data_1{local_domains_1.front(), fields_1["multi_field_1"]};
    cpu_double_data_descriptor_t serial_data_2{local_domains_2.front(), fields_2["serial_field_2"]};
    cpu_double_data_descriptor_t multi_data_2{local_domains_2.front(), fields_2["multi_field_2"]};

    // Serial halo exchange
    auto h_s1 = co.exchange(patterns_1(serial_data_1));
    h_s1.wait();
    auto h_s2 = co.exchange(patterns_2(serial_data_2));
    h_s2.wait();

    // Multiple halo exchange
    auto h_m = co.exchange(patterns_1(multi_data_1), patterns_2(multi_data_2));
    h_m.wait();

    // Test for correctness
    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            EXPECT_TRUE(serial_field_1_data(node, level) == multi_field_1_data(node, level));
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            EXPECT_TRUE(serial_field_2_data(node, level) == multi_field_2_data(node, level));
        }
    }

#ifdef __CUDACC__
    // Additional data descriptor types for GPU
    using gpu_int_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::gpu, domain_id_t, int>;
    using gpu_double_data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::gpu, domain_id_t, double>;

    // Additional fields for GPU halo exchange
    fields_1.add(fs_nodes_1.createField<int>(atlas::option::name("gpu_multi_field_1")));
    fields_2.add(fs_nodes_2.createField<double>(atlas::option::name("gpu_multi_field_2")));
    auto gpu_multi_field_1_data = atlas::array::make_host_view<int, 2>(fields_1["gpu_multi_field_1"]);
    auto gpu_multi_field_2_data = atlas::array::make_host_view<double, 2>(fields_2["gpu_multi_field_2"]);
    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            gpu_multi_field_1_data(node, level) = value;
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            auto value = ((rank << 15) + (node << 7) + level) * 0.5;
            gpu_multi_field_2_data(node, level) = value;
        }
    }
    fields_1["gpu_multi_field_1"].cloneToDevice();
    fields_2["gpu_multi_field_2"].cloneToDevice();

    // Additional data descriptors for GPU halo exchange
    gpu_int_data_descriptor_t gpu_multi_data_1{local_domains_1.front(), 0, fields_1["gpu_multi_field_1"]};
    gpu_double_data_descriptor_t gpu_multi_data_2{local_domains_2.front(), 0, fields_2["gpu_multi_field_2"]};

    // Multiple halo exchange on the GPU
    auto h_m_gpu = co.exchange(patterns_1(gpu_multi_data_1), patterns_2(gpu_multi_data_2));
    h_m_gpu.wait();

    // Test for correctness
    fields_1["gpu_multi_field_1"].cloneFromDevice();
    fields_2["gpu_multi_field_2"].cloneFromDevice();
    fields_1["gpu_multi_field_1"].reactivateHostWriteViews();
    fields_2["gpu_multi_field_2"].reactivateHostWriteViews();
    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            EXPECT_TRUE(serial_field_1_data(node, level) == gpu_multi_field_1_data(node, level));
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            EXPECT_TRUE(serial_field_2_data(node, level) == gpu_multi_field_2_data(node, level));
        }
    }
#endif

}
