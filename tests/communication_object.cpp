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
#include <mpi.h>
#include "ghex_arch.hpp"
#include "ghex_protocol.hpp"
#include "communication_object.hpp"


TEST(communication_object, constructor) {

    using domain_id_t = int;

    class my_pattern {

    public:

        using iteration_space_t = std::pair<int, int>;
        using halo_t = std::pair<domain_id_t, std::vector<iteration_space_t>>;

    private:

        std::vector<halo_t> m_send_halos;
        std::vector<halo_t> m_receive_halos;

    public:

        my_pattern() :
            m_send_halos{ halo_t{ 1, { iteration_space_t{1, 2}, iteration_space_t{2, 3} } } },
            m_receive_halos{ halo_t{ 1, { iteration_space_t{0, 1}, iteration_space_t{3, 4} } } } {}

        const std::vector<halo_t>& send_halos() const { return m_send_halos; }
        const std::vector<halo_t>& receive_halos() const { return m_receive_halos; };

    };

    using communication_object_t = ghex::communication_object<domain_id_t, my_pattern, ghex::ghex_mpi, ghex::ghex_cpu>;

    my_pattern p{};

    EXPECT_NO_THROW(communication_object_t c{p});

}
