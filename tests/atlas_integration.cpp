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

//#include "atlas/parallel/mpi/mpi.h"
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
#include "../include/atlas_domain_descriptor.hpp"
#include "../include/communication_object.hpp"


/** @brief CPU data descriptor
 * WARN: so far full support to multiple vertical layers is still not provided*/
template <typename T, typename DomainDescriptor>
class my_data_desc {

    public:

        using index_t = typename DomainDescriptor::index_t;
        using Byte = unsigned char;

    private:

        const DomainDescriptor& m_domain;
        const int* m_partition_data; // WARN: this should not be needed with a richer iteration spaces
        atlas::array::ArrayView<T, 2> m_values;

    public:

        my_data_desc(const DomainDescriptor& domain,
                     const atlas::Field& field) :
            m_domain{domain},
            m_partition_data{atlas::array::make_view<int, 1>(domain.partition()).data()},
            m_values{atlas::array::make_view<T, 2>(field)} {}

        /** @brief data type size, mandatory*/
        std::size_t data_type_size() const {
            return sizeof (T);
        }

        /** @brief single access set function, not mandatory but used by the corresponding multiple access operator*/
        void set(const T& value, const index_t idx, const std::size_t level) {
            m_values(idx, level) = value;
        }

        /** @brief single access get function, not mandatory but used by the corresponding multiple access operator*/
        const T& get(const index_t idx, const std::size_t level) const {
            return m_values(idx, level);
        }

        /** @brief multiple access set function, needed by GHEX in order to perform the unpacking.
         * WARN: it could be more efficient if the iteration space includes also the indexes on this domain;
         * in order to do so, iteration space needs to include an additional set of indexes;
         * for now, the needed indices are retrieved by looping over the whole doamin,
         * and filtering out all the indices by those on the desired remote partition.
         * @tparam IterationSpace iteration space type
         * @param is iteration space which to loop through in order to retrieve the coordinates at which to set back the buffer values
         * @param buffer buffer with the data to be set back*/
        template <typename IterationSpace>
        void set(const IterationSpace& is, const Byte* buffer) {
            for (index_t idx : is.local_index()) {
                for (std::size_t level = 0; level < is.levels(); ++level) {
                    set(*(reinterpret_cast<const T*>(buffer)), idx, level);
                    buffer += sizeof(T);
                }
            }
        }

        /** @brief multiple access get function, needed by GHEX in order to perform the packing
         * @tparam IterationSpace iteration space type
         * @param is iteration space which to loop through in order to retrieve the coordinates at which to get the data
         * @param buffer buffer to be filled*/
        template <typename IterationSpace>
        void get(const IterationSpace& is, Byte* buffer) const {
            for (index_t idx : is.local_index()) {
                for (std::size_t level = 0; level < is.levels(); ++level) {
                    std::memcpy(buffer, &get(idx, level), sizeof(T));
                    buffer += sizeof(T);
                }
            }
        }

};


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

    // Using atlas communicator
    // int rank = static_cast<int>(atlas::mpi::comm().rank());
    // int size = ...
    // Using our communicator
    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};
    int rank = comm.rank();

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
        gridtools::atlas_domain_descriptor<int> _d(0,
                                                   rank,
                                                   mesh.nodes().partition(),
                                                   mesh.nodes().remote_index(),
                                                   nb_levels,
                                                   nb_nodes);
    );

    gridtools::atlas_domain_descriptor<int> d{0,
                                              rank,
                                              mesh.nodes().partition(),
                                              mesh.nodes().remote_index(),
                                              nb_levels,
                                              nb_nodes};

    if (rank == 0) {
        EXPECT_TRUE(d.first() == 0);
        EXPECT_TRUE(d.last() == 421);
    }

}


TEST(atlas_integration, halo_generator) {

    // Using atlas communicator
    // int rank = static_cast<int>(atlas::mpi::comm().rank());
    // int size = ...
    // Using our communicator
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

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

    // Instantiate domain descriptor with halo size = 1
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

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};

    // 1) test: halo generator exceptions
    EXPECT_NO_THROW(auto halos_ = hg(d););

}


TEST(atlas_integration, make_pattern) {

    // Using atlas communicator
    // int rank = static_cast<int>(atlas::mpi::comm().rank());
    // int size = ...
    // Using our communicator
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

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

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

    EXPECT_NO_THROW(auto patterns_ = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains););

}


TEST(atlas_integration, halo_exchange) {

    using domain_descriptor_t = gridtools::atlas_domain_descriptor<int>;

    // Using atlas communicator
    // int rank = static_cast<int>(atlas::mpi::comm().rank());
    // int size = ...
    // Using our communicator
    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};
    int rank = comm.rank();
    int size = comm.size();

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
    my_data_desc<int, domain_descriptor_t> data_1{local_domains.front(), fields["GHEX_field_1"]};

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
