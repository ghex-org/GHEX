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
#include <sys/time.h>
#include <sstream>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include <gtest/gtest.h>
#include "gtest_main_gpu_atlas.cpp"

#include <boost/mpi/communicator.hpp>

#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/field.h"
#include "atlas/array_fwd.h"
#include "atlas/output/Gmsh.h" // needed only for debug, should be removed later
#include "atlas/runtime/Log.h" // needed only for debug, should be removed later

#include "../include/protocol/mpi.hpp"
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

    struct timeval start_atlas;
    struct timeval stop_atlas;
    double lapse_time_atlas;
    struct timeval start_GHEX;
    struct timeval stop_GHEX;
    double lapse_time_GHEX;

#ifndef NDEBUG
    std::stringstream ss;
    ss << rank;
    std::string filename = "halo_exchange_int_size=12_rank=" + ss.str() + "_CUDA.txt";
    std::ofstream file(filename.c_str());
#endif

    const std::size_t n_iter = 100;

#ifdef __CUDACC__
    GT_CUDA_CHECK(cudaSetDevice(0));
    if (!rank) std::cout << "Set CUDA device 0\n";
#endif

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("O256");
    if (!rank) std::cout << "Generated structured grid\n";

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);
    if (!rank) std::cout << "Generated mesh associated to structured grid\n";

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));
    if (!rank) std::cout << "Generated functionspace with halo = 1, associated to mesh\n";

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
    std::vector<domain_descriptor_t> local_domains{};
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    domain_descriptor_t d{rank, rank, mesh.nodes().partition(), mesh.nodes().remote_index(), nb_levels, nb_nodes_1};
    if (!rank) std::cout << "Generated domain descriptor\n";
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};
    if (!rank) std::cout << "Generated halo generator\n";

    // Make patterns
    auto patterns = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains);
    if (!rank) std::cout << "Generated patterns\n";

    // Istantiate communication objects
    auto cos = gridtools::make_communication_object(patterns);
    if (!rank) std::cout << "Generated communicatin objects\n";

    // Fields creation and initialization

    auto GHEX_field_1 = fs_nodes.createField<int>(atlas::option::name("GHEX_field_1")); // WARN: why no conversion to array is needed?
    auto atlas_field_1 = fs_nodes.createField<int>(atlas::option::name("atlas_field_1")); // WARN: why no conversion to array is needed?
    if (!rank) std::cout << "Generated atlas Fields\n";

    auto GHEX_field_1_h = atlas::array::make_host_view<int, 2>(GHEX_field_1);
    auto atlas_field_1_h = atlas::array::make_host_view<int, 2>(atlas_field_1);
    if (!rank) std::cout << "Created host views\n";
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            GHEX_field_1_h(node, level) = value;
            atlas_field_1_h(node, level) = value;
        }
    }

    GHEX_field_1.cloneToDevice();
    atlas_field_1.cloneToDevice();
    if (!rank) std::cout << "Fields cloned to device\n";

    gridtools::atlas_data_descriptor_gpu<int, domain_descriptor_t> data_1{local_domains.front(), 0, GHEX_field_1};
    if (!rank) std::cout << "Created data descriptor\n";

    // ==================== GHEX halo exchange ====================

    auto h = cos.exchange(patterns(data_1));
    h.wait();

    if (!rank) std::cout << "GHEX halo exchange, step 1, completed, now benchmarking\n";

    cudaDeviceSynchronize();

    gettimeofday(&start_GHEX, nullptr);

    for (auto i = 0; i < n_iter; ++i) {
        auto h_ = cos.exchange(patterns(data_1));
        h_.wait();
    }

    gettimeofday(&stop_GHEX, nullptr);

    cudaDeviceSynchronize();

    if (!rank) std::cout << "GHEX halo exchange completed\n";

    GHEX_field_1.cloneFromDevice();
    GHEX_field_1.reactivateHostWriteViews();

    lapse_time_GHEX =
        ((static_cast<double>(stop_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop_GHEX.tv_usec)) -
         (static_cast<double>(start_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_GHEX.tv_usec))) *
        1000.0 / n_iter;

    // ==================== atlas halo exchange ====================

    fs_nodes.haloExchange(atlas_field_1, true);

    if (!rank) std::cout << "atlas halo exchange, step 1, completed, now benchmarking\n";

    cudaDeviceSynchronize();

    gettimeofday(&start_atlas, nullptr);

    for (auto i = 0; i < n_iter; ++i) {
        fs_nodes.haloExchange(atlas_field_1, true);
    }

    gettimeofday(&stop_atlas, nullptr);

    cudaDeviceSynchronize();

    if (!rank) std::cout << "atlas halo exchange completed\n";

    atlas_field_1.cloneFromDevice();
    atlas_field_1.reactivateHostWriteViews();

    lapse_time_atlas =
        ((static_cast<double>(stop_atlas.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop_atlas.tv_usec)) -
         (static_cast<double>(start_atlas.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_atlas.tv_usec))) *
        1000.0 / n_iter;

    // ==================== test for correctness ====================

    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            EXPECT_TRUE(GHEX_field_1_h(node, level) == atlas_field_1_h(node, level));
#ifndef NDEBUG
            // Write output to file for comparing results with multiple node runs
            if (size == 12) {
                file << GHEX_field_1_h(node, level);
                if (GHEX_field_1_h(node, level) != atlas_field_1_h(node, level)) file << " INVALID VALUE";
                file << "\n";
            }
#endif
        }
    }

    std::cout << "rank " << rank << ", GHEX time = " << lapse_time_GHEX << ", atlas time = " << lapse_time_atlas << "\n";

}
