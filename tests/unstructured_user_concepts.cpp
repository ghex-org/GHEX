/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */

#include <set>
#include <vector>
#include <utility>

#include <gtest/gtest.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/arch_list.hpp>
#include <ghex/communication_object_2.hpp>


#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif
using context_type = gridtools::ghex::tl::context<transport, threading>;


/** @brief Test domain descriptor and halo generator concepts */
TEST(unstructured_user_concepts, domain_descriptor) {

    using domain_id_type = int;
    using global_index_type = int;
    using domain_descriptor_type = gridtools::ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
    using vertices_type = std::vector<global_index_type>;
    using vertices_set_type = std::set<global_index_type>;
    using adjncy_type = std::vector<global_index_type>;
    using map_type = std::vector<std::pair<global_index_type, adjncy_type>>;

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    // Explicit initialization without linking to ParMetis
    switch (rank) {
        case 0: {
            map_type v_map{
                std::make_pair(global_index_type{0},  adjncy_type{13, 2, 1, 20, 11}),
                std::make_pair(global_index_type{13}, adjncy_type{0,  5, 7}),
                std::make_pair(global_index_type{5},  adjncy_type{13, 2, 3}),
                std::make_pair(global_index_type{2},  adjncy_type{0,  5})
            };
            domain_descriptor_type d{0, v_map};
            EXPECT_TRUE(d.domain_id() == 0);
            EXPECT_TRUE(d.size() == 4);
            vertices_type reference_vertices{0, 13, 5, 2};
            EXPECT_TRUE(d.vertices() == reference_vertices);
            vertices_set_type reference_halo_vertices{1, 20, 7, 3, 11};
            vertices_set_type halo_vertices{d.halo_vertices().begin(), d.halo_vertices().end()};
            EXPECT_TRUE(halo_vertices == reference_halo_vertices);
            break;
        }
        case 1: {
            map_type v_map{
                std::make_pair(global_index_type{1},  adjncy_type{0,  19, 20, 7, 16}),
                std::make_pair(global_index_type{19}, adjncy_type{1,  4,  15, 8}),
                std::make_pair(global_index_type{20}, adjncy_type{0,  1,  4,  7}),
                std::make_pair(global_index_type{4},  adjncy_type{19, 20, 15, 8, 9}),
                std::make_pair(global_index_type{7},  adjncy_type{13, 1,  20, 15}),
                std::make_pair(global_index_type{15}, adjncy_type{19, 4,  7,  8}),
                std::make_pair(global_index_type{8},  adjncy_type{19, 4,  15})
            };
            domain_descriptor_type d{1, v_map};
            EXPECT_TRUE(d.domain_id() == 1);
            EXPECT_TRUE(d.size() == 7);
            vertices_type reference_vertices{1, 19, 20, 4, 7, 15, 8};
            EXPECT_TRUE(d.vertices() == reference_vertices);
            vertices_set_type reference_halo_vertices{0, 13, 16, 9};
            vertices_set_type halo_vertices{d.halo_vertices().begin(), d.halo_vertices().end()};
            EXPECT_TRUE(halo_vertices == reference_halo_vertices);
            break;
        }
        case 2: {
            map_type v_map{
                std::make_pair(global_index_type{3},  adjncy_type{5, 18, 6}),
                std::make_pair(global_index_type{16}, adjncy_type{1, 18}),
                std::make_pair(global_index_type{18}, adjncy_type{3, 16})
            };
            domain_descriptor_type d{2, v_map};
            EXPECT_TRUE(d.domain_id() == 2);
            EXPECT_TRUE(d.size() == 3);
            vertices_type reference_vertices{3, 16, 18};
            EXPECT_TRUE(d.vertices() == reference_vertices);
            vertices_set_type reference_halo_vertices{5, 1, 6};
            vertices_set_type halo_vertices{d.halo_vertices().begin(), d.halo_vertices().end()};
            EXPECT_TRUE(halo_vertices == reference_halo_vertices);
            break;
        }
        case 3: {
            map_type v_map{
                std::make_pair(global_index_type{17}, adjncy_type{11}),
                std::make_pair(global_index_type{6},  adjncy_type{3, 11, 10, 9}),
                std::make_pair(global_index_type{11}, adjncy_type{0, 17, 6, 10, 12}),
                std::make_pair(global_index_type{10}, adjncy_type{6, 11, 9}),
                std::make_pair(global_index_type{12}, adjncy_type{11, 9}),
                std::make_pair(global_index_type{9},  adjncy_type{4, 6, 10, 12})
            };
            domain_descriptor_type d{3, v_map};
            EXPECT_TRUE(d.domain_id() == 3);
            EXPECT_TRUE(d.size() == 6);
            vertices_type reference_vertices{17, 6, 11, 10, 12, 9};
            EXPECT_TRUE(d.vertices() == reference_vertices);
            vertices_set_type reference_halo_vertices{0, 4, 3};
            vertices_set_type halo_vertices{d.halo_vertices().begin(), d.halo_vertices().end()};
            EXPECT_TRUE(halo_vertices == reference_halo_vertices);
            break;
        }
    }

}


/** @brief Test data descriptor concept*/
/*TEST(unstructured_user_concepts, data_descriptor) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();
    int size = context.size();

}*/
