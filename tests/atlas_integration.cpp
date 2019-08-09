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

        /** @brief single access set function, not mandatory but used by the corresponding multiple access operator.
         * WARN: sets the whole column to the same value, full support to multiple vertical layers is not provided yet*/
        void set(const T& value, const index_t idx) {
            for (std::size_t level = 0; level < m_domain.levels(); ++level) {
                m_values(idx, level) = value;
            }
        }

        /** @brief single access get function, not mandatory but used by the corresponding multiple access operator.
         * WARN: returns the value of the first layer, full support to multiple vertical layers is not provided yet*/
        const T& get(const index_t idx) const {
            return m_values(idx, 0);
        }

        /** @brief multiple access set function, needed by GHEX in order to perform the unpacking.
         * WARN: it could be more efficient if the iteration space includes also the indexes on this domain;
         * in order to do so, iteration space needs to include an additional set of indices;
         * for now, the needed indices are retrieved by looping over the whole doamin,
         * and filtering out all the indices by those on the desired remote partition;
         * For now, the iteration space is used only to retrieve the right partition index.
         * @tparam IterationSpace iteration space type
         * @param is iteration space which to loop through in order to retrieve the coordinates at which to set back the buffer values
         * @param buffer buffer with the data to be set back*/
        template <typename IterationSpace>
        void set(const IterationSpace& is, const Byte* buffer) {
            if (is.partition() == m_domain.rank()) {
                for (index_t idx : is.remote_index()) {
                    set(*(reinterpret_cast<const T*>(buffer)), idx);
                    buffer += sizeof(T);
                }
            } else {
                for (index_t idx = 0; idx < m_domain.size(); ++idx) {
                    if (m_partition_data[idx] == is.partition()) { // WARN: needed static cast here?
                        set(*(reinterpret_cast<const T*>(buffer)), idx);
                        buffer += sizeof(T);
                    }
                }
            }
        }

        /** @brief multiple access get function, needed by GHEX in order to perform the packing
         * @tparam IterationSpace iteration space type
         * @param is iteration space which to loop through in order to retrieve the coordinates at which to get the data
         * @param buffer buffer to be filled*/
        template <typename IterationSpace>
        void get(const IterationSpace& is, Byte* buffer) const {
            for (index_t idx : is.remote_index()) {
                std::memcpy(buffer, &get(idx), sizeof(T));
                buffer += sizeof(T);
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

    // Field creation and initialization
    atlas::FieldSet fields;
    fields.add(fs_nodes.createField<double>(atlas::option::name("field_1")));
    auto field_1_data = atlas::array::make_view<double, 2>(fields["field_1"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            field_1_data(node, level) = static_cast<double>(rank);
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
    my_data_desc<double, domain_descriptor_t> data_1{local_domains.front(), fields["field_1"]};

    // Exchange
    auto h = cos.front().exchange(data_1);
    h.wait();

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
