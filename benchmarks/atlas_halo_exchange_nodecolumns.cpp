/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <gtest/gtest.h>

#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/functionspace.h"
#include "atlas/field.h"
#include "atlas/array.h"

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/threads/std_thread/primitives.hpp>
#include "../include/ghex/unstructured/grid.hpp"
#include "../include/ghex/unstructured/pattern.hpp"
#include "../include/ghex/glue/atlas/atlas_user_concepts.hpp"
#include "../include/ghex/communication_object_2.hpp"

#ifdef __CUDACC__
#include "gridtools/common/cuda_util.hpp"
#endif

#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif
using context_type = gridtools::ghex::tl::context<transport, threading>;


TEST(atlas_integration, halo_exchange_nodecolumns) {

    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<int>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

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
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    domain_descriptor_t d{rank,
                          rank,
                          mesh.nodes().partition(),
                          mesh.nodes().remote_index(),
                          nb_levels,
                          nb_nodes_1};
    local_domains.push_back(d);

    // Instantiate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{rank, size};

    // Make patterns
    using grid_type = gridtools::ghex::unstructured::grid;
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

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
    gridtools::ghex::atlas_data_descriptor<int, domain_descriptor_t> data_1{local_domains.front(), fields["GHEX_field_1"]};

    // Atlas halo exchange
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
    // Additional fields for GPU halo exchange
    fields.add(fs_nodes.createField<int>(atlas::option::name("atlas_field_1_gpu")));
    fields.add(fs_nodes.createField<int>(atlas::option::name("GHEX_field_1_gpu")));
    auto atlas_field_1_gpu_data = atlas::array::make_host_view<int, 2>(fields["atlas_field_1_gpu"]);
    auto GHEX_field_1_gpu_data = atlas::array::make_host_view<int, 2>(fields["GHEX_field_1_gpu"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            atlas_field_1_gpu_data(node, level) = value;
            GHEX_field_1_gpu_data(node, level) = value;
        }
    }
    fields["atlas_field_1_gpu"].cloneToDevice();
    fields["GHEX_field_1_gpu"].cloneToDevice();

    // Additional data descriptor for GPU halo exchange
    gridtools::ghex::atlas_data_descriptor_gpu<int, domain_descriptor_t> data_1_gpu{local_domains.front(), 0, fields["GHEX_field_1_gpu"]};

    // Atlas halo exchange
    fs_nodes.haloExchange(fields["atlas_field_1_gpu"], true);

    // GHEX halo exchange on GPU
    auto h_gpu = co.exchange(patterns(data_1_gpu));
    h_gpu.wait();

    // Test for correctness
    fields["atlas_field_1_gpu"].cloneFromDevice();
    fields["GHEX_field_1_gpu"].cloneFromDevice();
    fields["atlas_field_1_gpu"].reactivateHostWriteViews();
    fields["GHEX_field_1_gpu"].reactivateHostWriteViews();
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_data(node, level));
            // Not strictly needed, just double check Atlas on GPU
            EXPECT_TRUE(GHEX_field_1_gpu_data(node, level) == atlas_field_1_gpu_data(node, level));
        }
    }
#endif

}
