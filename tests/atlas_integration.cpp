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


/* CPU data descriptor */
template <typename T, typename DomainDescriptor>
class my_data_desc {

    using coordinate_t = typename DomainDescriptor::coordinate_type;
    using layout_map_t = gridtools::layout_map<0, 1, 2>;
    using Byte = unsigned char;

    const DomainDescriptor& m_domain;
    coordinate_t m_halos_offset;
    std::vector<T> m_values;

public:

    my_data_desc(const DomainDescriptor& domain,
                 const coordinate_t& halos_offset,
                 const std::vector<T>& values) :
        m_domain{domain},
        m_halos_offset{halos_offset},
        m_values{values} {}

    /** @brief data type size, mandatory*/
    std::size_t data_type_size() const {
        return sizeof (T);
    }

    /** @brief single access set function, not mandatory but used by the corresponding multiple access operator*/
    void set(const T& value, const coordinate_t& coords) {
        m_values[coords] = value;
    }

    /** @brief single access get function, not mandatory but used by the corresponding multiple access operator*/
    const T& get(const coordinate_t& coords) const {
        return m_values[coords];
    }

    /** @brief multiple access set function, needed by GHEX in order to perform the unpacking
     * @tparam IterationSpace iteration space type
     * @param is iteration space which to loop through in order to retrieve the coordinates at which to set back the buffer values
     * @param buffer buffer with the data to be set back*/
    template <typename IterationSpace>
    void set(const IterationSpace& is, const Byte* buffer) {
        gridtools::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            set(*(reinterpret_cast<const T*>(buffer)), coords);
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
    }

    /** @brief multiple access get function, needed by GHEX in order to perform the packing
     * @tparam IterationSpace iteration space type
     * @param is iteration space which to loop through in order to retrieve the coordinates at which to get the data
     * @param buffer buffer to be filled*/
    template <typename IterationSpace>
    void get(const IterationSpace& is, Byte* buffer) const {
        gridtools::detail::for_loop<3, 3, layout_map_t>::apply([this, &buffer](auto... indices){
            coordinate_t coords{indices...};
            const T* tmp_ptr{&get(coords)};
            std::memcpy(buffer, tmp_ptr, sizeof(T));
            buffer += sizeof(T);
        }, is.local().first(), is.local().last());
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
                                                   mesh.nodes().partition(),
                                                   mesh.nodes().remote_index(),
                                                   nb_nodes,
                                                   rank);
    );

    gridtools::atlas_domain_descriptor<int> d{0,
                                              mesh.nodes().partition(),
                                              mesh.nodes().remote_index(),
                                              nb_nodes,
                                              rank};

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
                                              mesh.nodes().partition(),
                                              mesh.nodes().remote_index(),
                                              nb_nodes_1,
                                              rank};

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
                                              mesh.nodes().partition(),
                                              mesh.nodes().remote_index(),
                                              nb_nodes_1,
                                              rank};
    local_domains.push_back(d);

    // Instantate halo generator
    gridtools::atlas_halo_generator<int> hg{rank, size};

    EXPECT_NO_THROW(auto patterns_ = gridtools::make_pattern<gridtools::unstructured_grid>(world, hg, local_domains););

}


TEST(atlas_integration, data_descriptor) {

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
                                              mesh.nodes().partition(),
                                              mesh.nodes().remote_index(),
                                              nb_nodes_1,
                                              rank};
    local_domains.push_back(d);

    // Following tests not needed, just a template code to play with fields

    // Few initial debugging info
    if (rank == 0) {
        std::cout << "Metadatafor rank 0: " << mesh.metadata() << "\n";
        std::cout << "number of nodes for functionspace, rank 0: " << fs_nodes.nb_nodes() << "\n";
    }

    // Field creation and initialization
    atlas::FieldSet fields;
    // Field field_scalar2 = fs_nodes.createField<double>( option::name( "scalar2" ) );
    fields.add(fs_nodes.createField<double>(atlas::option::name("temperature")));
    auto temperature = atlas::array::make_view<double, 2>(fields["temperature"]);
    for (auto node = 0; node < fs_nodes.nb_nodes(); ++node) {
        for (auto level = 0; level < fs_nodes.levels(); ++level) {
            temperature(node, level) = static_cast<double>(rank);
        }
    }

    // Write mesh and field in gmsh format before halo exchange (step 0)
    atlas::output::Gmsh gmsh_0("temperature_step_0.msh");
    gmsh_0.write(mesh);
    gmsh_0.write(fields["temperature"]);

    // Halo exchange
    fs_nodes.haloExchange(fields["temperature"]);

    // Write mesh and field in gmsh format after halo exchange (step 1)
    atlas::output::Gmsh gmsh_1("temperature_step_1.msh");
    gmsh_1.write(mesh);
    gmsh_1.write(fields["temperature"]);

    // Write final checksum
    std::string checksum = fs_nodes.checksum(fields["temperature"]);
    atlas::Log::info() << checksum << std::endl;

}
