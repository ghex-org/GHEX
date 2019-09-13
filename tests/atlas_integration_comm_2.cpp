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
#include "gtest_main_atlas.cpp"

#include <boost/mpi/communicator.hpp>

#include <gridtools/common/layout_map.hpp>

#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/field.h"
#include "atlas/array/ArrayView.h"
#include "atlas/output/Gmsh.h" // needed only for debug, should be removed later
#include "atlas/runtime/Log.h" // needed only for debug, should be removed later

#include "../include/protocol/mpi.hpp"
#include "../include/utils.hpp"
#include "../include/unstructured_grid.hpp"
#include "../include/unstructured_pattern.hpp"
#include "../include/atlas_user_concepts.hpp"
#include "../include/communication_object_2.hpp"


TEST(atlas_integration, halo_exchange) {

    using domain_descriptor_t = gridtools::atlas_domain_descriptor<int>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};
    int rank = comm.rank();
    int size = comm.size();

#ifndef NDEBUG
    std::stringstream ss;
    ss << rank;
    std::string filename = "halo_exchange_int_size=12_rank=" + ss.str() + ".txt";
    std::ofstream file(filename.c_str());
#endif

    // ==================== Atlas code ====================

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N16");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

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

    // ==================== GHEX code ====================

    // Instantiate vector of local domains
    std::vector<domain_descriptor_t> local_domains{};

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
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

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};

    // Make patterns
    auto patterns = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains);

    // Istantiate communication objects
    auto cos = gridtools::make_communication_object(patterns);

    // Istantiate data descriptor
    gridtools::atlas_data_descriptor<int, domain_descriptor_t> data_1{local_domains.front(), fields["GHEX_field_1"]};

    // ==================== atlas halo exchange ====================

    fs_nodes.haloExchange(fields["atlas_field_1"]);

    // ==================== GHEX halo exchange ====================

    auto h = cos.exchange(patterns(data_1));
    h.wait();

    // ==================== test for correctness ====================

    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
#ifndef NDEBUG
            // Write output to file for comparing results with multiple node runs
            if (size == 12) {
                file << GHEX_field_1_data(node, level);
                if (GHEX_field_1_data(node, level) != atlas_field_1_data(node, level)) file << " INVALID VALUE";
                file << "\n";
            }
#endif
        }
    }

}


TEST(atlas_integration, halo_exchange_multiple_patterns) {

    using domain_descriptor_t = gridtools::atlas_domain_descriptor<int>;

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};
    int rank = comm.rank();
    int size = comm.size();

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N16");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Instantiate 2 vector of local domains, respectively for halo size = 1 and halo size = 2
    std::vector<domain_descriptor_t> local_domains_1{};
    std::vector<domain_descriptor_t> local_domains_2{};

    // Generate 2 functionspaces associated to mesh and GHEX domain descriptors, accordingly
    atlas::functionspace::NodeColumns fs_nodes_1(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    domain_descriptor_t d_1{rank,
                            rank,
                            mesh.nodes().partition(),
                            mesh.nodes().remote_index(),
                            nb_levels,
                            nb_nodes_1};
    local_domains_1.push_back(d_1);
    atlas::functionspace::NodeColumns fs_nodes_2(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(2));
    std::stringstream ss_2;
    atlas::idx_t nb_nodes_2;
    ss_2 << "nb_nodes_including_halo[" << 2 << "]";
    mesh.metadata().get( ss_2.str(), nb_nodes_2 );
    domain_descriptor_t d_2{rank,
                            rank,
                            mesh.nodes().partition(),
                            mesh.nodes().remote_index(),
                            nb_levels,
                            nb_nodes_2};
    local_domains_2.push_back(d_2);

    // Fields creation and initialization
    atlas::FieldSet fields_1, fields_2;
    fields_1.add(fs_nodes_1.createField<int>(atlas::option::name("atlas_field_1")));
    fields_1.add(fs_nodes_1.createField<int>(atlas::option::name("GHEX_field_1")));
    fields_2.add(fs_nodes_2.createField<double>(atlas::option::name("atlas_field_2")));
    fields_2.add(fs_nodes_2.createField<double>(atlas::option::name("GHEX_field_2")));
    auto atlas_field_1_data = atlas::array::make_view<int, 2>(fields_1["atlas_field_1"]);
    auto GHEX_field_1_data = atlas::array::make_view<int, 2>(fields_1["GHEX_field_1"]);
    auto atlas_field_2_data = atlas::array::make_view<double, 2>(fields_2["atlas_field_2"]);
    auto GHEX_field_2_data = atlas::array::make_view<double, 2>(fields_2["GHEX_field_2"]);
    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            atlas_field_1_data(node, level) = value;
            GHEX_field_1_data(node, level) = value;
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            auto value = ((rank << 15) + (node << 7) + level) * 0.5;
            atlas_field_2_data(node, level) = value;
            GHEX_field_2_data(node, level) = value;
        }
    }

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};

    // Make patterns
    auto patterns_1 = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains_1);
    auto patterns_2 = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains_2);

    // Istantiate communication objects
    auto cos = gridtools::make_communication_object(patterns_1, patterns_2);

    // Istantiate data descriptors
    gridtools::atlas_data_descriptor<int, domain_descriptor_t> data_1{local_domains_1.front(), fields_1["GHEX_field_1"]};
    gridtools::atlas_data_descriptor<double, domain_descriptor_t> data_2{local_domains_2.front(), fields_2["GHEX_field_2"]};

    // ==================== atlas halo exchange ====================

    fs_nodes_1.haloExchange(fields_1["atlas_field_1"]);
    fs_nodes_2.haloExchange(fields_2["atlas_field_2"]);

    // ==================== GHEX halo exchange ====================

    auto h = cos.exchange(patterns_1(data_1), patterns_2(data_2));
    h.wait();

    // ==================== test for correctness ====================

    for (auto node = 0; node < fs_nodes_1.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_1.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
        }
    }
    for (auto node = 0; node < fs_nodes_2.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes_2.levels(); ++level) {
            EXPECT_DOUBLE_EQ(GHEX_field_2_data(node, level), atlas_field_2_data(node, level));
        }
    }

}
