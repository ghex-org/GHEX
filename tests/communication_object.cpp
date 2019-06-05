/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>
#include "gtest_main_boost.cpp"
#include <vector>
#include <array>
#include <utility>
#include <boost/mpi/communicator.hpp>
#include <gridtools/common/layout_map.hpp>
#include "ghex_arch.hpp"
#include "communication_object.hpp"
#include "../include/protocol/communicator_base.hpp"
#include "../include/protocol/mpi.hpp"
#include "triplet.hpp"
#include "utils.hpp"


class my_domain_desc {

public:

    using Coordinate = std::array<int, 3>;
    using DomainId = int;

private:

    DomainId m_id;
    Coordinate m_first;
    Coordinate m_last;

public:

    class halo {

        Coordinate m_first;
        Coordinate m_last;

    public:

        const Coordinate& first() const { return m_first; }
        const Coordinate& last() const { return m_last; }

    };

    DomainId domain_id() const { return m_id; }
    const Coordinate& first() const { return m_first; }
    const Coordinate& last() const { return m_last; }

};


/* CPU data descriptor */
template <typename T, typename DomainDescriptor>
class my_data_desc {

    using Coordinate = typename DomainDescriptor::Coordinate;
    using Layoutmap = gridtools::layout_map<2, 1, 0>;

    const DomainDescriptor& m_domain;
    array<triple_t<USE_DOUBLE, double>, Layoutmap> m_values;

public:

    using Byte = unsigned char;

    my_data_desc(const DomainDescriptor& domain, const array<triple_t<USE_DOUBLE, double>, Layoutmap>& values) :
        m_domain{domain},
        m_values{values} {}

    void store(const T& value, const Coordinate& coords) {
        m_values(coords[0], coords[1], coords[2]) = value;
    }

    const T* load(const Coordinate& coords) const {
        return m_values(coords[0], coords[1], coords[2]);
    }

    template <typename IterationSpace>
    void store(const IterationSpace& is, const Byte* buffer) {
        gridtools::detail::for_loop<3, 3, Layoutmap>::apply([this, &buffer](auto... indices){
            Coordinate c{indices...};
            store(static_cast<T>(buffer), c);
            buffer += sizeof(T);
        }, is.first(), is.last());
    }

    template <typename IterationSpace>
    void load(const IterationSpace& is, Byte* buffer) const {
        gridtools::detail::for_loop<3, 3, Layoutmap>::apply([this, &buffer](auto... indices){
            Coordinate c{indices...};
            buffer = static_cast<Byte>(load(c));
            buffer += sizeof(T);
        }, is.first(), is.last());
    }

};


TEST(communication_object, constructor) {

    boost::mpi::communicator world;
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

    // EXPECT_NO_THROW(communication_object_t c{p});

}
