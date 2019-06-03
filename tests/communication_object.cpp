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
#include <vector>
#include <array>
#include "ghex_arch.hpp"
#include "communication_object.hpp"


class my_domain_desc {

public:

    using coordinate_type = std::array<int, 3>;
    using domain_id_type = int;

private:

    domain_id_type m_id;
    coordinate_type m_first;
    coordinate_type m_last;

public:

    class halo {

        coordinate_type m_first;
        coordinate_type m_last;

    public:

        const coordinate_type& first() const { return m_first; }
        const coordinate_type& last() const { return m_last; }

    };

    domain_id_type domain_id() const { return m_id; }
    const coordinate_type& first() const { return m_first; }
    const coordinate_type& last() const { return m_last; }

};


class my_data_desc {};


TEST(communication_object, constructor) {

    // boost::mpi::communicator world;
    // gridtools::protocol::communicator<gridtools::protocol::mpi> comm{world};

    // ...

    // EXPECT_NO_THROW(communication_object_t c{p});

}
