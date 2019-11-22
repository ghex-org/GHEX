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

#include <mpi.h>

#include "atlas/library/Library.h"
#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/field.h"
#include "atlas/array/ArrayView.h"
#include "atlas/array_fwd.h"
#include "atlas/output/Gmsh.h" // needed only for debug, should be removed later
#include "atlas/runtime/Log.h" // needed only for debug, should be removed later

#include "../include/ghex/transport_layer/mpi/communicator_base.hpp"
#include "../include/ghex/transport_layer/communicator.hpp"
#include "../include/ghex/unstructured/grid.hpp"
#include "../include/ghex/unstructured/pattern.hpp"
#include "../include/ghex/glue/atlas/atlas_user_concepts.hpp"
#include "../include/ghex/communication_object_2.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "../include/ghex/cuda_utils/error.hpp"
#endif


int main(int argc, char** argv) {

    using domain_descriptor_t = gridtools::ghex::atlas_domain_descriptor<int>;

    MPI_Init(&argc, &argv);

    atlas::Library::instance().initialise(argc, argv);

    gridtools::ghex::tl::mpi::communicator_base mpi_comm;
    gridtools::ghex::tl::communicator<gridtools::ghex::tl::mpi_tag> comm{mpi_comm};
    int rank = comm.rank();
    int size = comm.size();

    struct timeval start_GHEX;
    struct timeval stop_GHEX;
    double lapse_time_GHEX;

    const std::size_t n_iter = 100;

#ifdef __CUDACC__
    GT_CUDA_CHECK(cudaSetDevice(0));
#endif

    // Generate Octahedral Gaussian grid
    atlas::StructuredGrid grid("O1280");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    // Number of vertical levels required
    std::size_t nb_levels = 100;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // ==================== Atlas code for CPU ====================

    // Fields creation and initialization
    auto atlas_field_1 = fs_nodes.createField<int>(atlas::option::name("atlas_field_1"));
    auto atlas_field_1_data = atlas::array::make_view<int, 2>(atlas_field_1);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            auto value = (rank << 15) + (node << 7) + level;
            atlas_field_1_data(node, level) = value;
        }
    }

    // Halo exchange on the CPU with Atlas
    fs_nodes.haloExchange(atlas_field_1);

    // ==================== GHEX code for GPU =====================

    // Instantiate domain descriptor with halo size = 1 and add it to local domains
    std::vector<domain_descriptor_t> local_domains{};
    std::stringstream ss_1;
    atlas::idx_t nb_nodes_1;
    ss_1 << "nb_nodes_including_halo[" << 1 << "]";
    mesh.metadata().get(ss_1.str(), nb_nodes_1);
    domain_descriptor_t d{rank, rank, mesh.nodes().partition(), mesh.nodes().remote_index(), nb_levels, nb_nodes_1};
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::ghex::atlas_halo_generator<int> hg{rank, size};

    // Make patterns
    auto patterns = gridtools::ghex::make_pattern<gridtools::ghex::unstructured::grid>(mpi_comm, hg, local_domains);

    // Istantiate communication objects
    auto cos = gridtools::ghex::make_communication_object<decltype(patterns)>();

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

    // Instantiate data descriptor
    gridtools::ghex::atlas_data_descriptor_gpu<int, domain_descriptor_t> data_1{local_domains.front(), 0, GHEX_field_1};

    // Halo exchange on the GPU with GHEX
    auto h = cos.exchange(patterns(data_1));
    h.wait();

    // ==================== GHEX benchmarks on the GPU ============

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

    // ==================== Test for correctness ==================

    bool passed = true;
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            passed = passed && (GHEX_field_1_h(node, level) == atlas_field_1_data(node, level));
        }
    }

    if (passed) {
        std::cout << "rank " << rank << ", result: PASSED, GHEX time = " << lapse_time_GHEX << "\n";
    } else {
        std::cout << "rank " << rank << ", result: FAILED, GHEX time = " << lapse_time_GHEX << "\n";
    }

    atlas::Library::instance().finalise();

    MPI_Finalize();

}
