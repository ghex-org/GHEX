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

    struct timeval start_GHEX;
    struct timeval stop_GHEX;
    double lapse_time_GHEX;

    std::stringstream ss;
    ss << rank;
    std::string filename = "halo_exchange_int_size=12_rank=" + ss.str() + ".txt";
    std::ifstream file(filename.c_str());
    bool file_reading_mode;
    file_reading_mode = (size == 12) && file.good();
    if (file_reading_mode) {
        std::cout << "Running actual test reading from file " << filename << "\n";
    } else {
        std::cout << "WARN: file reading mode not available, running halo exchange without checking results";
    }

    const std::size_t n_iter = 100;

#ifdef __CUDACC__
    GT_CUDA_CHECK(cudaSetDevice(0));
#endif

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("O256");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
    std::vector<domain_descriptor_t> local_domains{};
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get( ss_1.str(), nb_nodes_1 );
    domain_descriptor_t d{rank, rank, mesh.nodes().partition(), mesh.nodes().remote_index(), nb_levels, nb_nodes_1};
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};

    // Make patterns
    auto patterns = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains);

    // Istantiate communication objects
    auto cos = gridtools::make_communication_object(patterns);

    // Fields creation and initialization

    auto GHEX_field_1 = fs_nodes.createField<int>(atlas::option::name("GHEX_field_1")); // WARN: why no conversion to array is needed?

    auto GHEX_field_1_h = atlas::array::make_host_view<int, 2>(GHEX_field_1);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            GHEX_field_1_h(node, level) = value;
        }
    }

    GHEX_field_1.cloneToDevice();

    gridtools::atlas_data_descriptor_gpu<int, domain_descriptor_t> data_1{local_domains.front(), 0, GHEX_field_1};

    // ==================== GHEX halo exchange ====================

    auto h = cos.exchange(patterns(data_1));
    h.wait();

    cudaDeviceSynchronize();

    gettimeofday(&start_GHEX, nullptr);

    for (auto i = 0; i < n_iter; ++i) {
        auto h_ = cos.exchange(patterns(data_1));
        h_.wait();
    }

    gettimeofday(&stop_GHEX, nullptr);

    cudaDeviceSynchronize();

    GHEX_field_1.cloneFromDevice();
    GHEX_field_1.reactivateHostWriteViews();

    lapse_time_GHEX =
        ((static_cast<double>(stop_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(stop_GHEX.tv_usec)) -
         (static_cast<double>(start_GHEX.tv_sec) + 1 / 1000000.0 * static_cast<double>(start_GHEX.tv_usec))) *
        1000.0 / n_iter;

    // ==================== test for correctness ====================

    if (file_reading_mode) {
        int value;
        for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
            for (auto level = 0; level < fs_nodes.levels(); ++level) {
                file >> value;
                EXPECT_TRUE(GHEX_field_1_h(node, level) == value);
            }
        }
    }

    std::cout << "rank " << rank << ", GHEX time = " << lapse_time_GHEX << "\n";

}
