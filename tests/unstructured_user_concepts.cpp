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

#include <vector>
#include <thread>

#include <gtest/gtest.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/unstructured/communication_object_ipr.hpp>
#include "./util/unstructured_test_case.hpp"


using data_descriptor_cpu_int_type = gridtools::ghex::unstructured::data_descriptor<gridtools::ghex::cpu, domain_id_type, global_index_type, int>;

#ifndef GHEX_TEST_UNSTRUCTURED_OVERSUBSCRIPTION

/** @brief Test domain descriptor and halo generator concepts */
TEST(unstructured_user_concepts, domain_descriptor_and_halos) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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

    // setup patterns using recv_domain_ids_gen
    recv_domain_ids_gen<> rdig{};
    auto patterns_d_ids = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

    // check halos
    check_send_halos_indices(patterns_d_ids[0]);
    check_recv_halos_indices(patterns_d_ids[0]);

}

/** @brief Test data descriptor concept*/
TEST(unstructured_user_concepts, data_descriptor) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(context.get_communicator());

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

/** @brief Test in place receive*/
TEST(unstructured_user_concepts, in_place_receive) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id{rank}; // 1 domain per rank
    domain_descriptor_type d{domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)};
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(context.get_communicator());

    // application data
    std::vector<int> field(d.size(), 0);
    initialize_data(d, field);
    data_descriptor_cpu_int_type data{d, field};

    auto h = co.exchange(patterns(data));
    h.wait();

    // check exchanged data
    check_exchanged_data(d, field, patterns[0]);

}

/** @brief Test in place receive with multiple fields*/
TEST(unstructured_user_concepts, in_place_receive_multi) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id{rank}; // 1 domain per rank
    domain_descriptor_type d{domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)};
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(context.get_communicator());

    // application data

    std::vector<int> field_1(d.size(), 0);
    initialize_data(d, field_1);
    data_descriptor_cpu_int_type data_1{d, field_1};

    std::vector<int> field_2(d.size(), 0);
    initialize_data(d, field_2);
    data_descriptor_cpu_int_type data_2{d, field_2};

    std::vector<int> field_3(d.size(), 0);
    initialize_data(d, field_3);
    data_descriptor_cpu_int_type data_3{d, field_3};

    auto h = co.exchange(patterns(data_1), patterns(data_2), patterns(data_3));
    h.wait();

    // check exchanged data
    check_exchanged_data(d, field_1, patterns[0]);
    check_exchanged_data(d, field_2, patterns[0]);
    check_exchanged_data(d, field_3, patterns[0]);

}

#else

#ifndef GHEX_TEST_UNSTRUCTURED_THREADS

/** @brief Test pattern setup with multiple domains per rank */
TEST(unstructured_user_concepts, pattern_setup_oversubscribe) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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

    // setup patterns using recv_domain_ids_gen
    auto domain_to_rank = [](const domain_id_type d_id){ return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns_d_ids = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

}

/** @brief Test pattern setup with multiple domains per rank, oddly distributed */
TEST(unstructured_user_concepts, pattern_setup_oversubscribe_asymm) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    halo_generator_type hg{};
    auto domain_to_rank = [](const domain_id_type d_id){ return (d_id != 3) ? int{0} : int{1}; };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};

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

            // setup patterns
            auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);
            check_send_halos_indices(patterns[1]);
            check_recv_halos_indices(patterns[1]);
            check_send_halos_indices(patterns[2]);
            check_recv_halos_indices(patterns[2]);

            // setup patterns using recv_domain_ids_gen
            auto patterns_d_ids = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

            // check halos
            check_send_halos_indices(patterns_d_ids[0]);
            check_recv_halos_indices(patterns_d_ids[0]);
            check_send_halos_indices(patterns_d_ids[1]);
            check_recv_halos_indices(patterns_d_ids[1]);
            check_send_halos_indices(patterns_d_ids[2]);
            check_recv_halos_indices(patterns_d_ids[2]);

            break;

        }

        case 1: {

            domain_id_type domain_id_1{3};
            auto v_map_1 = init_v_map(domain_id_1);
            domain_descriptor_type d_1{domain_id_1, v_map_1};
            std::vector<domain_descriptor_type> local_domains{d_1};

            // setup patterns
            auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);

            // setup patterns using recv_domain_ids_gen
            auto patterns_d_ids = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

            // check halos
            check_send_halos_indices(patterns_d_ids[0]);
            check_recv_halos_indices(patterns_d_ids[0]);

            break;

        }

    }

}

/** @brief Test data descriptor concept*/
TEST(unstructured_user_concepts, data_descriptor_oversubscribe) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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

    auto domain_to_rank = [](const domain_id_type d_id){ return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(context.get_communicator());

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

/** @brief Test data descriptor concept with in-place receive*/
TEST(unstructured_user_concepts, data_descriptor_oversubscribe_ipr) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id_1{rank * 2};
    domain_id_type domain_id_2{rank * 2 + 1}; // HERE domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)
    domain_descriptor_type d_1{domain_id_1, init_ordered_vertices(domain_id_1), init_inner_sizes(domain_id_1)};
    domain_descriptor_type d_2{domain_id_2, init_ordered_vertices(domain_id_2), init_inner_sizes(domain_id_2)};
    std::vector<domain_descriptor_type> local_domains{d_1, d_2};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(context.get_communicator());

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

#else

/** @brief Test data descriptor concept with multiple threads*/
TEST(unstructured_user_concepts, data_descriptor_oversubscribe_threads) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
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

    auto domain_to_rank = [](const domain_id_type d_id){ return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, rdig, local_domains);
    using pattern_container_type = decltype(patterns);

    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    auto func = [&context](auto bi) {
        auto co = gridtools::ghex::make_communication_object<pattern_container_type>(context.get_communicator());
        auto h = co.exchange(bi);
        h.wait();
    };

    std::vector<std::thread> threads;
    threads.push_back(std::thread{func, patterns(data_1)});
    threads.push_back(std::thread{func, patterns(data_2)});
    for (auto& t : threads) t.join();

    // check exchanged data
    check_exchanged_data(d_1, field_1, patterns[0]);
    check_exchanged_data(d_2, field_2, patterns[1]);

}

/** @brief Test data descriptor concept with in-place receive and multiple threads*/
TEST(unstructured_user_concepts, data_descriptor_oversubscribe_ipr_threads) {

    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int rank = context.rank();

    domain_id_type domain_id_1{rank * 2};
    domain_id_type domain_id_2{rank * 2 + 1}; // HERE domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)
    domain_descriptor_type d_1{domain_id_1, init_ordered_vertices(domain_id_1), init_inner_sizes(domain_id_1)};
    domain_descriptor_type d_2{domain_id_2, init_ordered_vertices(domain_id_2), init_inner_sizes(domain_id_2)};
    std::vector<domain_descriptor_type> local_domains{d_1, d_2};
    halo_generator_type hg{};

    auto patterns = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);
    using pattern_container_type = decltype(patterns);

    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    auto func = [&context](auto bi) {
        auto co = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(context.get_communicator());
        auto h = co.exchange(bi);
        h.wait();
    };

    std::vector<std::thread> threads;
    threads.push_back(std::thread{func, patterns(data_1)});
    threads.push_back(std::thread{func, patterns(data_2)});
    for (auto& t : threads) t.join();

    // check exchanged data
    check_exchanged_data(d_1, field_1, patterns[0]);
    check_exchanged_data(d_2, field_2, patterns[1]);

}

#endif // GHEX_TEST_UNSTRUCTURED_THREADS

#endif // GHEX_TEST_UNSTRUCTURED_OVERSUBSCRIPTION
