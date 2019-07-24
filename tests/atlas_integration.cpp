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
#include <gridtools/common/layout_map.hpp>

#include <gtest/gtest.h>
#include "gtest_main_atlas.cpp"

#include "atlas/grid.h"
#include "atlas/mesh.h"
#include "atlas/meshgenerator.h"
#include "atlas/functionspace/NodeColumns.h"
#include "atlas/output/Gmsh.h"

#include "../include/utils.hpp"
#include "../include/unstructured_pattern.hpp"


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


TEST(atlas_integration, halo_exchange) {

    // Generate global classic reduced Gaussian grid
    atlas::StructuredGrid grid("N32");

    // Generate mesh associated to structured grid
    atlas::StructuredMeshGenerator meshgenerator;
    atlas::Mesh mesh = meshgenerator.generate(grid);

    atlas::output::Gmsh gmsh("N32.msh");
    gmsh.write(mesh);

    // Number of vertical levels required
    std::size_t nb_levels = 10;

    // Generate functionspace associated to mesh
    atlas::functionspace::NodeColumns fs_nodes(mesh, atlas::option::levels(nb_levels) | atlas::option::halo(1));

}
