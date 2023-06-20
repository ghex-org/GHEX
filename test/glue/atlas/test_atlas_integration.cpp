/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>
#include "../../mpi_runner/mpi_test_fixture.hpp"

#include <ghex/config.hpp>
#if defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST)
#include <gridtools/storage/cpu_kfirst.hpp>
#elif defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST)
#include <gridtools/storage/cpu_ifirst.hpp>
#endif
#ifdef GHEX_CUDACC
#include <gridtools/storage/gpu.hpp>
#include <gridtools/common/cuda_util.hpp>
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/meshgenerator.h>
#include <atlas/functionspace.h>
#include <atlas/field.h>
#include <atlas/array.h>

#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/glue/atlas/field.hpp>
#include <ghex/glue/atlas/atlas_user_concepts.hpp>
#include <ghex/communication_object.hpp>

#include <vector>

TEST_F(mpi_test_fixture, atlas_halo_exchange)
{
    using domain_id_t = int;
    using domain_descriptor_t = ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = ghex::unstructured::grid;
#if defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_kfirst;
#elif defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_ifirst;
#endif
    using function_space_t = atlas::functionspace::NodeColumns;
    using cpu_data_descriptor_t = ghex::atlas_data_descriptor<ghex::cpu, domain_id_t, int,
        storage_traits_cpu, function_space_t>;

    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};
    int           rank = ctxt.rank();

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O256");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh                    mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 10;

    // Generate functionspace associated to the mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh,
        atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor
    std::vector<domain_descriptor_t> local_domains{};
    domain_descriptor_t              d{rank, mesh.nodes().partition(), mesh.nodes().remote_index()};
    local_domains.push_back(d);

    // Instantiate halo generator
    ghex::atlas_halo_generator<int> hg{};

    // Instantiate recv domain ids generator
    ghex::atlas_recv_domain_ids_gen<int> rdig{};

    // Make patterns
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);

    // Make communication object
    auto co = ghex::make_communication_object<decltype(patterns)>(ctxt);

    // Fields creation and initialization
    auto atlas_field_1 = fs_nodes.createField<int>(atlas::option::name("atlas_field_1"));
    auto GHEX_field_1 = ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes, nb_levels);
    {
        auto atlas_field_1_data = atlas::array::make_view<int, 2>(atlas_field_1);
        auto GHEX_field_1_data = GHEX_field_1.host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes.levels(); ++level)
            {
                auto value = (rank << 15) + (node << 7) + level;
                atlas_field_1_data(node, level) = value;
                GHEX_field_1_data(node, level) = value;
            }
        }
    }

    // GHEX target view
    auto GHEX_field_1_target_data = GHEX_field_1.target_view();

    // Instantiate data descriptor
    cpu_data_descriptor_t data_1{local_domains.front(), GHEX_field_1_target_data,
        GHEX_field_1.components()};

    // Atlas halo exchange (reference values)
    fs_nodes.haloExchange(atlas_field_1);

    // GHEX halo exchange
    auto h = co.exchange(patterns(data_1));
    h.wait();

    // test for correctness
    {
        auto atlas_field_1_data = atlas::array::make_view<const int, 2>(atlas_field_1);
        auto GHEX_field_1_data = GHEX_field_1.const_host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes.levels(); ++level)
            {
                EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
            }
        }
    }

#ifdef GHEX_CUDACC

    using storage_traits_gpu = gridtools::storage::gpu;

    // Additional data descriptor type for GPU
    using gpu_data_descriptor_t = ghex::atlas_data_descriptor<ghex::gpu, domain_id_t, int,
        storage_traits_gpu, function_space_t>;

    // Additional field for GPU halo exchange
    auto GHEX_field_1_gpu = ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes, nb_levels);
    {
        auto GHEX_field_1_gpu_data = GHEX_field_1_gpu.host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes.levels(); ++level)
            {
                auto value = (rank << 15) + (node << 7) + level;
                GHEX_field_1_gpu_data(node, level) = value;
            }
        }
    }

    // GHEX target view
    auto GHEX_field_1_gpu_target_data = GHEX_field_1_gpu.target_view();

    // Additional data descriptor for GPU halo exchange
    gpu_data_descriptor_t data_1_gpu{local_domains.front(), 0, GHEX_field_1_gpu_target_data,
        GHEX_field_1_gpu.components()};

    // GHEX halo exchange on GPU
    auto h_gpu = co.exchange(patterns(data_1_gpu));
    h_gpu.wait();

    // Test for correctness
    {
        auto atlas_field_1_data = atlas::array::make_view<const int, 2>(atlas_field_1);
        auto GHEX_field_1_gpu_data = GHEX_field_1_gpu.const_host_view();
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes.levels(); ++level)
            {
                EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_data(node, level));
            }
        }
    }

#endif
}

TEST_F(mpi_test_fixture, atlas_halo_exchange_multiple_patterns)
{
    using domain_id_t = int;
    using domain_descriptor_t = ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = ghex::unstructured::grid;
#if defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_KFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_kfirst;
#elif defined(GHEX_ATLAS_GT_STORAGE_CPU_BACKEND_IFIRST)
    using storage_traits_cpu = gridtools::storage::cpu_ifirst;
#endif
    using function_space_t = atlas::functionspace::NodeColumns;
    using cpu_int_data_descriptor_t = ghex::atlas_data_descriptor<ghex::cpu, domain_id_t, int,
        storage_traits_cpu, function_space_t>;
    using cpu_double_data_descriptor_t = ghex::atlas_data_descriptor<ghex::cpu, domain_id_t, double,
        storage_traits_cpu, function_space_t>;

    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};
    int           rank = ctxt.rank();

    // Global octahedral Gaussian grid
    atlas::StructuredGrid grid("O256");

    // Generate mesh
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh                    mesh = meshgenerator.generate(grid);

    // Number of vertical levels
    std::size_t nb_levels = 10;

    // Generate functionspace associated to the mesh with halo size = 1
    atlas::functionspace::NodeColumns fs_nodes_1(mesh,
        atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor (halo size = 1)
    std::vector<domain_descriptor_t> local_domains_1{};
    domain_descriptor_t d_1{rank, mesh.nodes().partition(), mesh.nodes().remote_index()};
    local_domains_1.push_back(d_1);

    // Generate functionspace associated to the mesh with halo size = 2
    atlas::functionspace::NodeColumns fs_nodes_2(mesh,
        atlas::option::levels(nb_levels) | atlas::option::halo(2));

    // Instantiate domain descriptor (halo size = 2)
    std::vector<domain_descriptor_t> local_domains_2{};
    domain_descriptor_t d_2{rank, mesh.nodes().partition(), mesh.nodes().remote_index()};
    local_domains_2.push_back(d_2);

    // Instantate halo generator
    ghex::atlas_halo_generator<int> hg{};

    // Instantiate recv domain ids generator
    ghex::atlas_recv_domain_ids_gen<int> rdig{};

    // Make patterns
    auto patterns_1 = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains_1);
    auto patterns_2 = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains_2);

    // Make communication object
    auto co = ghex::make_communication_object<decltype(patterns_1)>(ctxt);

    // Fields creation and initialization
    auto serial_field_1 = ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes_1, nb_levels);
    auto multi_field_1 = ghex::atlas::make_field<int, storage_traits_cpu>(fs_nodes_1, nb_levels);
    auto serial_field_2 =
        ghex::atlas::make_field<double, storage_traits_cpu>(fs_nodes_2, nb_levels);
    auto multi_field_2 = ghex::atlas::make_field<double, storage_traits_cpu>(fs_nodes_2, nb_levels);
    {
        auto serial_field_1_data = serial_field_1.host_view();
        auto multi_field_1_data = multi_field_1.host_view();
        auto serial_field_2_data = serial_field_2.host_view();
        auto multi_field_2_data = multi_field_2.host_view();
        for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_1.levels(); ++level)
            {
                auto value = (rank << 15) + (node << 7) + level;
                serial_field_1_data(node, level) = value;
                multi_field_1_data(node, level) = value;
            }
        }
        for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_2.levels(); ++level)
            {
                auto value = ((rank << 15) + (node << 7) + level) * 0.5;
                serial_field_2_data(node, level) = value;
                multi_field_2_data(node, level) = value;
            }
        }
    }

    // GHEX target views
    auto serial_field_1_target_data = serial_field_1.target_view();
    auto multi_field_1_target_data = multi_field_1.target_view();
    auto serial_field_2_target_data = serial_field_2.target_view();
    auto multi_field_2_target_data = multi_field_2.target_view();

    // Instantiate data descriptors
    cpu_int_data_descriptor_t    serial_data_1{local_domains_1.front(), serial_field_1_target_data,
        serial_field_1.components()};
    cpu_int_data_descriptor_t    multi_data_1{local_domains_1.front(), multi_field_1_target_data,
        multi_field_1.components()};
    cpu_double_data_descriptor_t serial_data_2{local_domains_2.front(), serial_field_2_target_data,
        serial_field_2.components()};
    cpu_double_data_descriptor_t multi_data_2{local_domains_2.front(), multi_field_2_target_data,
        multi_field_2.components()};

    // Serial halo exchange
    auto h_s1 = co.exchange(patterns_1(serial_data_1));
    h_s1.wait();
    auto h_s2 = co.exchange(patterns_2(serial_data_2));
    h_s2.wait();

    // Multiple halo exchange
    auto h_m = co.exchange(patterns_1(multi_data_1), patterns_2(multi_data_2));
    h_m.wait();

    // Test for correctness
    {
        auto serial_field_1_data = serial_field_1.const_host_view();
        auto multi_field_1_data = multi_field_1.const_host_view();
        auto serial_field_2_data = serial_field_2.const_host_view();
        auto multi_field_2_data = multi_field_2.const_host_view();
        for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_1.levels(); ++level)
            {
                EXPECT_TRUE(serial_field_1_data(node, level) == multi_field_1_data(node, level));
            }
        }
        for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_2.levels(); ++level)
            {
                EXPECT_TRUE(serial_field_2_data(node, level) ==
                            multi_field_2_data(node,
                                level)); // TO DO: hard-coded 3d view. Should be more flexible
            }
        }
    }

#ifdef GHEX_CUDACC

    using storage_traits_gpu = gridtools::storage::gpu;

    // Additional data descriptor types for GPU
    using gpu_int_data_descriptor_t = ghex::atlas_data_descriptor<ghex::gpu, domain_id_t, int,
        storage_traits_gpu, function_space_t>;
    using gpu_double_data_descriptor_t = ghex::atlas_data_descriptor<ghex::gpu, domain_id_t, double,
        storage_traits_gpu, function_space_t>;

    // Additional fields for GPU halo exchange
    auto gpu_multi_field_1 =
        ghex::atlas::make_field<int, storage_traits_gpu>(fs_nodes_1, nb_levels);
    auto gpu_multi_field_2 =
        ghex::atlas::make_field<double, storage_traits_gpu>(fs_nodes_2, nb_levels);
    {
        auto gpu_multi_field_1_data = gpu_multi_field_1.host_view();
        auto gpu_multi_field_2_data = gpu_multi_field_2.host_view();
        for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_1.levels(); ++level)
            {
                auto value = (rank << 15) + (node << 7) + level;
                gpu_multi_field_1_data(node, level) = value;
            }
        }
        for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_2.levels(); ++level)
            {
                auto value = ((rank << 15) + (node << 7) + level) * 0.5;
                gpu_multi_field_2_data(node, level) = value;
            }
        }
    }

    // GHEX target views
    auto gpu_multi_field_1_target_data = gpu_multi_field_1.target_view();
    auto gpu_multi_field_2_target_data = gpu_multi_field_2.target_view();

    // Additional data descriptors for GPU halo exchange
    gpu_int_data_descriptor_t    gpu_multi_data_1{local_domains_1.front(), 0,
        gpu_multi_field_1_target_data, gpu_multi_field_1.components()};
    gpu_double_data_descriptor_t gpu_multi_data_2{local_domains_2.front(), 0,
        gpu_multi_field_2_target_data, gpu_multi_field_2.components()};

    // Multiple halo exchange on the GPU
    auto h_m_gpu = co.exchange(patterns_1(gpu_multi_data_1), patterns_2(gpu_multi_data_2));
    h_m_gpu.wait();

    // Test for correctness
    {
        auto serial_field_1_data = serial_field_1.const_host_view();
        auto gpu_multi_field_1_data = gpu_multi_field_1.const_host_view();
        auto serial_field_2_data = serial_field_2.const_host_view();
        auto gpu_multi_field_2_data = gpu_multi_field_2.const_host_view();
        for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_1.levels(); ++level)
            {
                EXPECT_TRUE(
                    serial_field_1_data(node, level) == gpu_multi_field_1_data(node, level));
            }
        }
        for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node)
        {
            for (auto level = 0; level < fs_nodes_2.levels(); ++level)
            {
                EXPECT_TRUE(
                    serial_field_2_data(node, level) == gpu_multi_field_2_data(node, level));
            }
        }
    }

#endif
}
