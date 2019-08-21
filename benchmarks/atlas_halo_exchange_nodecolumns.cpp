/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>
#include <sys/time.h>

#include <boost/mpi/communicator.hpp>

//#include "atlas/parallel/mpi/mpi.h"
#include "atlas/library/Library.h"
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
#include "../include/communication_object.hpp"


int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);

    atlas::Library::instance().initialise(argc, argv);

    using domain_descriptor_t = gridtools::atlas_domain_descriptor<int>;

    // Using atlas communicator
    // int rank = static_cast<int>(atlas::mpi::comm().rank());
    // int size = ...
    // Using our communicator
    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};
    int rank = comm.rank();
    int size = comm.size();

    struct timeval start_atlas;
    struct timeval stop_atlas;
    double lapse_time_atlas;
    struct timeval start_GHEX;
    struct timeval stop_GHEX;
    double lapse_time_GHEX;

    const std::size_t n_iter = 100;

    // ==================== Atlas code ====================

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("O1280");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 100;

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
            atlas_field_1_data(node, level) = rank;
            GHEX_field_1_data(node, level) = rank;
        }
    }

    // ==================== GHEX code ====================

    // Instantiate vector of local domains
    std::vector<gridtools::atlas_domain_descriptor<int>> local_domains{};

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    gridtools::atlas_domain_descriptor<int> d{0,
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

    // Istantiate communication object
    using communication_object_t = gridtools::communication_object<decltype(patterns)::value_type, gridtools::cpu>;
    std::vector<communication_object_t> cos;
    for (const auto& p : patterns) {
        cos.push_back(communication_object_t{p});
    }

    // Istantiate data descriptor
    gridtools::atlas_data_descriptor<int, domain_descriptor_t> data_1{local_domains.front(), fields["GHEX_field_1"]};

    // ==================== atlas halo exchange ====================

    MPI_Barrier(world);

    fs_nodes.haloExchange(fields["atlas_field_1"]);

    MPI_Barrier(world);

    gettimeofday(&start_atlas, nullptr);

    for (auto i = 0; i < n_iter; ++i) {
        fs_nodes.haloExchange(fields["atlas_field_1"]);
    }

    gettimeofday(&stop_atlas, nullptr);

    MPI_Barrier(world);

    lapse_time_atlas =
            ((static_cast<double>(stop_atlas.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop_atlas.tv_usec)) -
             (static_cast<double>(start_atlas.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_atlas.tv_usec))) *
            1000.0 / n_iter;

    // ==================== GHEX halo exchange ====================

    MPI_Barrier(world);

    auto h = cos.front().exchange(data_1);
    h.wait();

    MPI_Barrier(world);

    gettimeofday(&start_GHEX, nullptr);

    for (auto i = 0; i < n_iter; ++i) {
        auto h_ = cos.front().exchange(data_1);
        h_.wait();
    }

    gettimeofday(&stop_GHEX, nullptr);

    MPI_Barrier(world);

    lapse_time_GHEX =
            ((static_cast<double>(stop_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop_GHEX.tv_usec)) -
             (static_cast<double>(start_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_GHEX.tv_usec))) *
            1000.0 / n_iter;

    // ==================== test for correctness ====================

    bool passed = true;

    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            passed = passed and (GHEX_field_1_data(node, level) == atlas_field_1_data(node, level));
        }
    }

    if (passed) {
        std::cout << "RESULT: PASSED!\n";
        std::cout << "rank = " << rank << "; atlas time = " << lapse_time_atlas << "ms; GHEX time = " << lapse_time_GHEX << "ms\n";
    } else {
        std::cout << "RESULT: FAILED!\n";
    }

    atlas::Library::instance().finalise();

    MPI_Finalize();

    return 0;

}
