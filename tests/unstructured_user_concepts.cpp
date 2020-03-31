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
#include <map>
#include <utility>
#include <cassert>

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
using communicator_type = context_type::communicator_type;
using domain_id_type = int;
using global_index_type = int;
using domain_descriptor_type = gridtools::ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = gridtools::ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = gridtools::ghex::unstructured::grid;
using pattern_type = gridtools::ghex::pattern<communicator_type, grid_type::type<domain_descriptor_type>, domain_id_type>;
using local_indices_type = pattern_type::iteration_space::local_indices_type;
using vertices_type = domain_descriptor_type::vertices_type;
using vertices_set_type = domain_descriptor_type::vertices_set_type;
using adjncy_type = domain_descriptor_type::adjncy_type;
using map_type = domain_descriptor_type::map_type;
using it_diff_type = vertices_type::iterator::difference_type;
using data_descriptor_cpu_int_type = gridtools::ghex::unstructured::data_descriptor<gridtools::ghex::cpu, domain_id_type, global_index_type, int>;


/* Domains
 *
 *             id  |          inner           |        halo        |
 *              ----------------------------------------------------
 *              0  | [0, 13, 5, 2]            | [1, 3, 7, 11, 20]  |
 *              1  | [1, 19, 20, 4, 7, 15, 8] | [0, 9, 13, 16]     |
 *              2  | [3, 16, 18]              | [1, 5, 6]          |
 *              3  | [17, 6, 11, 10, 12, 9]   | [0, 3, 4]          |
 *
 * */


map_type init_v_map(const domain_id_type domain_id) {
    switch (domain_id) {
        case 0: {
            map_type v_map{std::make_pair(global_index_type{0},  adjncy_type{13, 2, 1, 20, 11}),
                        std::make_pair(global_index_type{13}, adjncy_type{0,  5, 7}),
                        std::make_pair(global_index_type{5},  adjncy_type{13, 2, 3}),
                        std::make_pair(global_index_type{2},  adjncy_type{0,  5})};
            return v_map;
        }
        case 1: {
            map_type v_map{std::make_pair(global_index_type{1},  adjncy_type{0,  19, 20, 7, 16}),
                        std::make_pair(global_index_type{19}, adjncy_type{1,  4,  15, 8}),
                        std::make_pair(global_index_type{20}, adjncy_type{0,  1,  4,  7}),
                        std::make_pair(global_index_type{4},  adjncy_type{19, 20, 15, 8, 9}),
                        std::make_pair(global_index_type{7},  adjncy_type{13, 1,  20, 15}),
                        std::make_pair(global_index_type{15}, adjncy_type{19, 4,  7,  8}),
                        std::make_pair(global_index_type{8},  adjncy_type{19, 4,  15})};
            return v_map;
        }
        case 2: {
            map_type v_map{std::make_pair(global_index_type{3},  adjncy_type{5, 18, 6}),
                        std::make_pair(global_index_type{16}, adjncy_type{1, 18}),
                        std::make_pair(global_index_type{18}, adjncy_type{3, 16})};
            return v_map;
        }
        case 3: {
            map_type v_map{std::make_pair(global_index_type{17}, adjncy_type{11}),
                        std::make_pair(global_index_type{6},  adjncy_type{3, 11, 10, 9}),
                        std::make_pair(global_index_type{11}, adjncy_type{0, 17, 6, 10, 12}),
                        std::make_pair(global_index_type{10}, adjncy_type{6, 11, 9}),
                        std::make_pair(global_index_type{12}, adjncy_type{11, 9}),
                        std::make_pair(global_index_type{9},  adjncy_type{4, 6, 10, 12})};
            return v_map;
        }
        default: {
            map_type v_map{};
            return v_map;
        }
    }
}


void check_domain(const domain_descriptor_type& d) {
    auto domain_id = d.domain_id();
    switch (domain_id) {
        case 0: {
            EXPECT_TRUE(d.inner_size() == 4);
            EXPECT_TRUE(d.size() == 9);
            vertices_type inner_vertices{d.vertices().begin(), d.vertices().begin() + static_cast<it_diff_type>(d.inner_size())};
            vertices_type reference_inner_vertices{0, 13, 5, 2};
            EXPECT_TRUE(inner_vertices == reference_inner_vertices);
            break;
        }
        case 1: {
            EXPECT_TRUE(d.inner_size() == 7);
            EXPECT_TRUE(d.size() == 11);
            vertices_type inner_vertices{d.vertices().begin(), d.vertices().begin() + static_cast<it_diff_type>(d.inner_size())};
            vertices_type reference_inner_vertices{1, 19, 20, 4, 7, 15, 8};
            EXPECT_TRUE(inner_vertices == reference_inner_vertices);
            break;
        }
        case 2: {
            EXPECT_TRUE(d.inner_size() == 3);
            EXPECT_TRUE(d.size() == 6);
            vertices_type inner_vertices{d.vertices().begin(), d.vertices().begin() + static_cast<it_diff_type>(d.inner_size())};
            vertices_type reference_inner_vertices{3, 16, 18};
            EXPECT_TRUE(inner_vertices == reference_inner_vertices);
            break;
        }
        case 3: {
            EXPECT_TRUE(d.inner_size() == 6);
            EXPECT_TRUE(d.size() == 9);
            vertices_type inner_vertices{d.vertices().begin(), d.vertices().begin() + static_cast<it_diff_type>(d.inner_size())};
            vertices_type reference_inner_vertices{17, 6, 11, 10, 12, 9};
            EXPECT_TRUE(inner_vertices == reference_inner_vertices);
            break;
        }
    }
}


void check_halo_generator(const domain_descriptor_type& d, const halo_generator_type& hg) {
    auto h = hg(d);
    switch (d.domain_id()) {
        case 0: {
            vertices_set_type halo_vertices_set{h.vertices().begin(), h.vertices().end()};
            vertices_set_type reference_halo_vertices_set{1, 20, 7, 3, 11};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 1: {
            vertices_set_type halo_vertices_set{h.vertices().begin(), h.vertices().end()};
            vertices_set_type reference_halo_vertices_set{0, 13, 16, 9};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 2: {
            vertices_set_type halo_vertices_set{h.vertices().begin(), h.vertices().end()};
            vertices_set_type reference_halo_vertices_set{5, 1, 6};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 3: {
            vertices_set_type halo_vertices_set{h.vertices().begin(), h.vertices().end()};
            vertices_set_type reference_halo_vertices_set{0, 4, 3};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
    }
}


/* Send maps (local indices on the send size)
 *
 *                             receivers
 *
 *             id  |     0     |     1     |     2     |     3     |
 *             -----------------------------------------------------
 *              0  |     -     |  [0, 1]   |    [2]    |    [0]    |
 *   senders    1  | [0, 4, 2] |     -     |    [0]    |    [3]    |
 *              2  |    [0]    |    [1]    |     -     |    [0]    |
 *              3  |    [2]    |    [5]    |    [1]    |     -     |
 *
 * */
void check_send_halos_indices(const pattern_type& p) {
    std::map<domain_id_type, local_indices_type> ref_map{};
    switch (p.domain_id()) {
        case 0: {
            ref_map.insert({std::make_pair(domain_id_type{1}, local_indices_type{0, 1}),
                            std::make_pair(domain_id_type{2}, local_indices_type{2}),
                            std::make_pair(domain_id_type{3}, local_indices_type{0})});
            break;
        }
        case 1: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{0, 4, 2}),
                            std::make_pair(domain_id_type{2}, local_indices_type{0}),
                            std::make_pair(domain_id_type{3}, local_indices_type{3})});
            break;
        }
        case 2: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{0}),
                            std::make_pair(domain_id_type{1}, local_indices_type{1}),
                            std::make_pair(domain_id_type{3}, local_indices_type{0})});
            break;
        }
        case 3: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{2}),
                            std::make_pair(domain_id_type{1}, local_indices_type{5}),
                            std::make_pair(domain_id_type{2}, local_indices_type{1})});
            break;
        }
    }
    EXPECT_TRUE(p.send_halos().size() == 3); // size is correct
    std::set<domain_id_type> res_ids{};
    for (const auto& sh : p.send_halos()) {
        auto res = res_ids.insert(sh.first.id);
        EXPECT_TRUE(res.second); // ids are unique
        EXPECT_NO_THROW(ref_map.at(sh.first.id)); // ids are correct
        EXPECT_TRUE(sh.second.front().local_indices() == ref_map.at(sh.first.id)); // indices are correct
    }
}


/* Recv maps (local indices on the recv side)
 *
 *                             receivers
 *
 *             id  |     0     |     1     |     2     |     3     |
 *             -----------------------------------------------------
 *              0  |     -     |  [7, 9]   |    [4]    |    [6]    |
 *   senders    1  | [4, 6, 8] |     -     |    [3]    |    [8]    |
 *              2  |    [5]    |   [10]    |     -     |    [7]    |
 *              3  |    [7]    |    [8]    |    [5]    |     -     |
 *
 * */
void check_recv_halos_indices(const pattern_type& p) {
    std::map<domain_id_type, local_indices_type> ref_map{};
    switch (p.domain_id()) {
        case 0: {
            ref_map.insert({std::make_pair(domain_id_type{1}, local_indices_type{4, 6, 8}),
                            std::make_pair(domain_id_type{2}, local_indices_type{5}),
                            std::make_pair(domain_id_type{3}, local_indices_type{7})});
            break;
        }
        case 1: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{7, 9}),
                            std::make_pair(domain_id_type{2}, local_indices_type{10}),
                            std::make_pair(domain_id_type{3}, local_indices_type{8})});
            break;
        }
        case 2: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{4}),
                            std::make_pair(domain_id_type{1}, local_indices_type{3}),
                            std::make_pair(domain_id_type{3}, local_indices_type{5})});
            break;
        }
        case 3: {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{6}),
                            std::make_pair(domain_id_type{1}, local_indices_type{8}),
                            std::make_pair(domain_id_type{2}, local_indices_type{7})});
            break;
        }
    }
    EXPECT_TRUE(p.recv_halos().size() == 3); // size is correct
    std::set<domain_id_type> res_ids{};
    for (const auto& rh : p.recv_halos()) {
        auto res = res_ids.insert(rh.first.id);
        EXPECT_TRUE(res.second); // ids are unique
        EXPECT_NO_THROW(ref_map.at(rh.first.id)); // ids are correct
        EXPECT_TRUE(rh.second.front().local_indices() == ref_map.at(rh.first.id)); // indices are correct
    }
}


template <typename Container>
void initialize_data(const domain_descriptor_type& d, Container& field) {
    using value_type = typename Container::value_type;
    assert(field.size() == d.size());
    for (std::size_t idx = 0; idx < d.inner_size(); ++idx) {
        field[idx] = static_cast<value_type>(d.domain_id()) * 100 + static_cast<value_type>(d.vertices()[idx]);
    }
}


template <typename Container>
void check_exchanged_data(const domain_descriptor_type& d, const Container& field, const pattern_type& p) {
    using value_type = typename Container::value_type;
    using index_type = pattern_type::index_type;
    std::map<index_type, domain_id_type> halo_map{};
    for (const auto& rh : p.recv_halos()) {
        for (const auto idx : rh.second.front().local_indices()) {
            halo_map.insert(std::make_pair(idx, rh.first.id));
        }
    }
    for (const auto& pair : halo_map) {
        EXPECT_EQ(field[static_cast<std::size_t>(pair.first)],
                static_cast<value_type>(pair.second * 100 + d.vertices()[static_cast<std::size_t>(pair.first)]));
    }
}


#ifndef GHEX_TEST_UNSTRUCTURED_OVERSUBSCRIPTION

/** @brief Test domain descriptor and halo generator concepts */
TEST(unstructured_user_concepts, domain_descriptor_and_halos) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    // domain
    domain_id_type domain_id{rank}; // 1 domain per rank
    auto v_map = init_v_map(domain_id);
    domain_descriptor_type d{domain_id, v_map};
    check_domain(d);

    // halo_generator
    halo_generator_type hg{};
    check_halo_generator(d, hg);

}

/** @brief Test pattern setup */
TEST(unstructured_user_concepts, pattern_setup) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id{rank}; // 1 domain per rank
    auto v_map = init_v_map(domain_id);
    domain_descriptor_type d{domain_id, v_map};
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};

    // setup patterns
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // check halos
    check_send_halos_indices(patterns[0]);
    check_recv_halos_indices(patterns[0]);

}

/** @brief Test data descriptor concept*/
TEST(unstructured_user_concepts, data_descriptor) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id{rank}; // 1 domain per rank
    auto v_map = init_v_map(domain_id);
    domain_descriptor_type d{domain_id, v_map};
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(context.get_communicator(context.get_token()));

    // application data
    std::vector<int> field(d.size(), 0);
    initialize_data(d, field);
    data_descriptor_cpu_int_type data{d, field};

    EXPECT_NO_THROW(co.bexchange(patterns(data)));

    auto h = co.exchange(patterns(data));
    h.wait();

    // check exchanged data
    check_exchanged_data(d, field, patterns[0]);

}

#else

/** @brief Test pattern setup with multiple domains per rank */
TEST(unstructured_user_concepts, pattern_setup_oversubscribe) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id_1{rank * 2};
    domain_id_type domain_id_2{rank * 2 + 1};
    auto v_map_1 = init_v_map(domain_id_1);
    auto v_map_2 = init_v_map(domain_id_2);
    domain_descriptor_type d_1{domain_id_1, v_map_1};
    domain_descriptor_type d_2{domain_id_2, v_map_2};
    std::vector<domain_descriptor_type> local_domains{d_1, d_2};
    halo_generator_type hg{};

    // setup patterns
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // check halos
    check_send_halos_indices(patterns[0]);
    check_recv_halos_indices(patterns[0]);
    check_send_halos_indices(patterns[1]);
    check_recv_halos_indices(patterns[1]);

}

/** @brief Test pattern setup with multiple domains per rank, oddly distributed */
TEST(unstructured_user_concepts, pattern_setup_oversubscribe_asymm) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    switch (rank) {

        case 0: {

            domain_id_type domain_id_1{0};
            domain_id_type domain_id_2{1};
            domain_id_type domain_id_3{2};
            auto v_map_1 = init_v_map(domain_id_1);
            auto v_map_2 = init_v_map(domain_id_2);
            auto v_map_3 = init_v_map(domain_id_3);
            domain_descriptor_type d_1{domain_id_1, v_map_1};
            domain_descriptor_type d_2{domain_id_2, v_map_2};
            domain_descriptor_type d_3{domain_id_3, v_map_3};
            std::vector<domain_descriptor_type> local_domains{d_1, d_2, d_3};
            halo_generator_type hg{};

            // setup patterns
            auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);
            check_send_halos_indices(patterns[1]);
            check_recv_halos_indices(patterns[1]);
            check_send_halos_indices(patterns[2]);
            check_recv_halos_indices(patterns[2]);

            break;

        }

        case 1: {

            domain_id_type domain_id_1{3};
            auto v_map_1 = init_v_map(domain_id_1);
            domain_descriptor_type d_1{domain_id_1, v_map_1};
            std::vector<domain_descriptor_type> local_domains{d_1};
            halo_generator_type hg{};

            // setup patterns
            auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);

            break;

        }

    }

}

/** @brief Test data descriptor concept*/
TEST(unstructured_user_concepts, data_descriptor_oversubscribe) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id_1{rank * 2};
    domain_id_type domain_id_2{rank * 2 + 1};
    auto v_map_1 = init_v_map(domain_id_1);
    auto v_map_2 = init_v_map(domain_id_2);
    domain_descriptor_type d_1{domain_id_1, v_map_1};
    domain_descriptor_type d_2{domain_id_2, v_map_2};
    std::vector<domain_descriptor_type> local_domains{d_1, d_2};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(context.get_communicator(context.get_token()));

    // application data
    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    EXPECT_NO_THROW(co.bexchange(patterns(data_1), patterns(data_2)));

    auto h = co.exchange(patterns(data_1), patterns(data_2));
    h.wait();

    // check exchanged data
    check_exchanged_data(d_1, field_1, patterns[0]);
    check_exchanged_data(d_2, field_2, patterns[1]);

}

#endif
