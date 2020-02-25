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
#include <ghex/communication_object.hpp>


#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif
using context_type = gridtools::ghex::tl::context<transport, threading>;


TEST(atlas_integration, dependencies) {

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N32");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    EXPECT_NO_THROW(
        atlas::functionspace::NodeColumns fs_nodes(mesh,
                                                   atlas::option::levels(nb_levels) | atlas::option::halo(1));
    );

}


TEST(atlas_integration, domain_descriptor) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N16");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    std::stringstream ss;
    atlas::idx_t nb_nodes;
    ss << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss.str(), nb_nodes );

    EXPECT_NO_THROW(
        gridtools::ghex::atlas_domain_descriptor<int> _d(0,
                                                         rank,
                                                         mesh.nodes().partition(),
                                                         mesh.nodes().remote_index(),
                                                         nb_levels,
                                                         nb_nodes);
    );

    gridtools::ghex::atlas_domain_descriptor<int> d{0,
                                                    rank,
                                                    mesh.nodes().partition(),
                                                    mesh.nodes().remote_index(),
                                                    nb_levels,
                                                    nb_nodes};

    if ((size == 4) && (rank == 0)) {
        EXPECT_TRUE(d.first() == 0);
        EXPECT_TRUE(d.last() == 421);
    }

}


TEST(atlas_integration, halo_generator) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N16");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor with halo size = 1
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    gridtools::ghex::atlas_domain_descriptor<int> d{0,
                                                    rank,
                                                    mesh.nodes().partition(),
                                                    mesh.nodes().remote_index(),
                                                    nb_levels,
                                                    nb_nodes_1};

    // Instantate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{size};

    // 1) test: halo generator exceptions
    EXPECT_NO_THROW(auto halos_ = hg(d););

}


TEST(atlas_integration, make_pattern) {

    using grid_type = gridtools::ghex::unstructured::grid;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N16");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate vector of local domains
    std::vector<gridtools::ghex::atlas_domain_descriptor<int>> local_domains{};

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    gridtools::ghex::atlas_domain_descriptor<int> d{0,
                                                    rank,
                                                    mesh.nodes().partition(),
                                                    mesh.nodes().remote_index(),
                                                    nb_levels,
                                                    nb_nodes_1};
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{size};

    EXPECT_NO_THROW(auto patterns_ = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains););

}


TEST(atlas_integration, halo_exchange) {

    using domain_id_t = int;
    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<domain_id_t>;
    using grid_type = gridtools::ghex::unstructured::grid;
    using data_descriptor_t = gridtools::ghex::atlas_data_descriptor<gridtools::ghex::cpu, domain_id_t, int>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

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
    domain_descriptor_t d{0,
                          rank,
                          mesh.nodes().partition(),
                          mesh.nodes().remote_index(),
                          nb_levels,
                          nb_nodes_1};
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{size};

    // Make patterns
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // Istantiate communication object
    using communication_object_t = gridtools::ghex::communication_object<decltype(patterns)::value_type, gridtools::ghex::cpu>;
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p, context.get_communicator(context.get_token())});
    }

    // Istantiate data descriptor
    data_descriptor_t data_1{local_domains.front(), fields["GHEX_field_1"]};

    // ==================== atlas halo exchange ====================

    fs_nodes.haloExchange(fields["atlas_field_1"]);

    // ==================== GHEX halo exchange ====================

    auto h = cos.front().exchange(data_1);
    h.wait();

    // ==================== test for correctness ====================

    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
        }
    }

    // ==================== Useful code snippets ====================

    // if (rank == 0) {
    //     std::cout << "Metadatafor rank 0: " << mesh.metadata() << "\n";
    //     std::cout << "number of nodes for functionspace, rank 0: " << fs_nodes.nb_nodes() << "\n";
    // }

    // Write mesh and field in gmsh format before halo exchange (step 0)
    // atlas::output::Gmsh gmsh_0("temperature_step_0.msh");
    // gmsh_0.write(mesh);
    // gmsh_0.write(fields["temperature"]);

    // Halo exchange
    // fs_nodes.haloExchange(fields["temperature"]);

    // Write mesh and field in gmsh format after halo exchange (step 1)
    // atlas::output::Gmsh gmsh_1("temperature_step_1.msh");
    // gmsh_1.write(mesh);
    // gmsh_1.write(fields["temperature"]);

    // Write final checksum
    // std::string checksum = fs_nodes.checksum(fields["temperature"]);
    // atlas::Log::info() << checksum << std::endl;

}
