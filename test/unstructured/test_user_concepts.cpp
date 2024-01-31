/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>
#include "../mpi_runner/mpi_test_fixture.hpp"

#include <ghex/config.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/communication_object.hpp>
#include <ghex/unstructured/communication_object_ipr.hpp>
#include "./unstructured_test_case.hpp"
#include "../util/memory.hpp"

#include <vector>
#include <thread>

using data_descriptor_cpu_int_type =
    ghex::unstructured::data_descriptor<ghex::cpu, domain_id_type, global_index_type, int>;
#ifdef GHEX_CUDACC
using data_descriptor_gpu_int_type =
    ghex::unstructured::data_descriptor<ghex::gpu, domain_id_type, global_index_type, int>;
#endif

void test_domain_descriptor_and_halos(ghex::context& ctxt);

void test_pattern_setup(ghex::context& ctxt);
void test_pattern_setup_oversubscribe(ghex::context& ctxt);
void test_pattern_setup_oversubscribe_asymm(ghex::context& ctxt);

void test_data_descriptor(ghex::context& ctxt, std::size_t levels, bool levels_first);
void test_data_descriptor_oversubscribe(ghex::context& ctxt);
void test_data_descriptor_threads(ghex::context& ctxt);

void test_in_place_receive(ghex::context& ctxt);
//void test_in_place_receive_multi(ghex::context& ctxt);
//void test_in_place_receive_oversubscribe(ghex::context& ctxt);
void test_in_place_receive_threads(ghex::context& ctxt);

TEST_F(mpi_test_fixture, domain_descriptor)
{
    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};

    if (world_size == 4) { test_domain_descriptor_and_halos(ctxt); }
}

TEST_F(mpi_test_fixture, pattern_setup)
{
    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};

    if (world_size == 4) { test_pattern_setup(ctxt); }
    else if (world_size == 2)
    {
        test_pattern_setup_oversubscribe(ctxt);
        test_pattern_setup_oversubscribe_asymm(ctxt);
    }
}

TEST_F(mpi_test_fixture, data_descriptor)
{
    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};

    if (world_size == 4)
    {
        test_data_descriptor(ctxt, 1, true);
        test_data_descriptor(ctxt, 3, true);
        test_data_descriptor(ctxt, 1, false);
        test_data_descriptor(ctxt, 3, false);
    }
    else if (world_size == 2)
    {
        test_data_descriptor_oversubscribe(ctxt);
        if (thread_safe) test_data_descriptor_threads(ctxt);
    }
}

TEST_F(mpi_test_fixture, in_place_receive)
{
    ghex::context ctxt{MPI_COMM_WORLD, thread_safe};

    if (world_size == 4)
    {
        test_in_place_receive(ctxt);
        //test_in_place_receive_multi(ctxt);
    }
    else if (world_size == 2)
    {
        //test_in_place_receive_oversubscribe(ctxt);
        if (thread_safe) test_in_place_receive_threads(ctxt);
    }
}

auto
create_halo(const domain_descriptor_type& d)
{
    // consider all outer vertices as halo
    auto halo = d.outer_gids();
    return halo;
}

halo_generator_type
make_halo_gen(const std::vector<domain_descriptor_type>& local_domains)
{
    std::set<global_index_type> halo_set;
    for (const auto& ld : local_domains)
    {
        for (const auto& x : ld.outer_gids()) halo_set.insert(x);
    }
    return {halo_set.begin(), halo_set.end()};
}

/** @brief Test domain descriptor and halo generator concepts */
void
test_domain_descriptor_and_halos(ghex::context& ctxt)
{
    // domain
    auto d = make_domain(ctxt.rank());
    check_domain(d);

    // halo_generator
    halo_generator_type hg{create_halo(d)};
    check_halo_generator(d, hg);
}

/** @brief Test pattern setup */
void
test_pattern_setup(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain(ctxt.rank())};

    // halo_generator
    auto hg = make_halo_gen(local_domains);

    // setup patterns
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

    // check halos
    check_send_halos_indices(patterns[0]);
    check_recv_halos_indices(patterns[0]);

    // setup patterns using recv_domain_ids_gen
    recv_domain_ids_gen<> rdig{};
    auto patterns_d_ids = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);

    // check halos
    check_send_halos_indices(patterns_d_ids[0]);
    check_recv_halos_indices(patterns_d_ids[0]);
}

/** @brief Test pattern setup with multiple domains per rank */
void
test_pattern_setup_oversubscribe(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain(ctxt.rank() * 2),
        make_domain(ctxt.rank() * 2 + 1)};

    // halo_generator
    auto hg = make_halo_gen(local_domains);

    // setup patterns
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

    // check halos
    check_send_halos_indices(patterns[0]);
    check_recv_halos_indices(patterns[0]);
    check_send_halos_indices(patterns[1]);
    check_recv_halos_indices(patterns[1]);

    // setup patterns using recv_domain_ids_gen
    auto domain_to_rank = [](const domain_id_type d_id) { return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns_d_ids = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);
}

/** @brief Test pattern setup with multiple domains per rank, oddly distributed */
void
test_pattern_setup_oversubscribe_asymm(ghex::context& ctxt)
{
    int rank = ctxt.rank();

    auto domain_to_rank = [](const domain_id_type d_id) { return (d_id != 3) ? int{0} : int{1}; };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};

    switch (rank)
    {
        case 0:
        {
            // domain
            std::vector<domain_descriptor_type> local_domains{make_domain(0), make_domain(1),
                make_domain(2)};

            // halo_generator
            auto hg = make_halo_gen(local_domains);

            // setup patterns
            auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);
            check_send_halos_indices(patterns[1]);
            check_recv_halos_indices(patterns[1]);
            check_send_halos_indices(patterns[2]);
            check_recv_halos_indices(patterns[2]);

            // setup patterns using recv_domain_ids_gen
            auto patterns_d_ids = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);

            // check halos
            check_send_halos_indices(patterns_d_ids[0]);
            check_recv_halos_indices(patterns_d_ids[0]);
            check_send_halos_indices(patterns_d_ids[1]);
            check_recv_halos_indices(patterns_d_ids[1]);
            check_send_halos_indices(patterns_d_ids[2]);
            check_recv_halos_indices(patterns_d_ids[2]);

            break;
        }

        case 1:
        {
            // domain
            std::vector<domain_descriptor_type> local_domains{make_domain(3)};

            // halo generator
            auto hg = make_halo_gen(local_domains);

            // setup patterns
            auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

            // check halos
            check_send_halos_indices(patterns[0]);
            check_recv_halos_indices(patterns[0]);

            // setup patterns using recv_domain_ids_gen
            auto patterns_d_ids = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);

            // check halos
            check_send_halos_indices(patterns_d_ids[0]);
            check_recv_halos_indices(patterns_d_ids[0]);

            break;
        }
    }
}

/** @brief Test data descriptor concept*/
void
test_data_descriptor(ghex::context& ctxt, std::size_t levels, bool levels_first)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain(ctxt.rank())};

    // halo generator
    auto hg = make_halo_gen(local_domains);

    // setup patterns
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = ghex::make_communication_object<pattern_container_type>(ctxt);

    // application data
    auto& d = local_domains[0];
    ghex::test::util::memory<int> field(d.size()*levels, 0);
    initialize_data(d, field, levels, levels_first);
    data_descriptor_cpu_int_type data{d, field, levels, levels_first};

    EXPECT_NO_THROW(co.exchange(patterns(data)).wait());

    auto h = co.exchange(patterns(data));
    h.wait();

    // check exchanged data
    check_exchanged_data(d, field, patterns[0], levels, levels_first);

#ifdef GHEX_CUDACC
    // application data
    initialize_data(d, field, levels, levels_first);
    field.clone_to_device();
    data_descriptor_gpu_int_type data_gpu{d, field.device_data(), levels, levels_first, 0, 0};

    EXPECT_NO_THROW(co.exchange(patterns(data_gpu)).wait());

    auto h_gpu = co.exchange(patterns(data_gpu));
    h_gpu.wait();

    // check exchanged data
    field.clone_to_host();
    check_exchanged_data(d, field, patterns[0], levels, levels_first);
#endif
}

/** @brief Test data descriptor concept*/
void
test_data_descriptor_oversubscribe(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain(ctxt.rank() * 2),
        make_domain(ctxt.rank() * 2 + 1)};

    // halo generator
    auto hg = make_halo_gen(local_domains);

    auto domain_to_rank = [](const domain_id_type d_id) { return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);

    // communication object
    using pattern_container_type = decltype(patterns);
    auto co = ghex::make_communication_object<pattern_container_type>(ctxt);

    // application data
    auto&            d_1 = local_domains[0];
    auto&            d_2 = local_domains[1];
    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    EXPECT_NO_THROW(co.exchange(patterns(data_1), patterns(data_2)).wait());

    auto h = co.exchange(patterns(data_1), patterns(data_2));
    h.wait();

    // check exchanged data
    check_exchanged_data(d_1, field_1, patterns[0]);
    check_exchanged_data(d_2, field_2, patterns[1]);
}

/** @brief Test data descriptor concept with multiple threads*/
void
test_data_descriptor_threads(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain(ctxt.rank() * 2),
        make_domain(ctxt.rank() * 2 + 1)};

    // halo generator
    auto hg = make_halo_gen(local_domains);

    auto domain_to_rank = [](const domain_id_type d_id) { return static_cast<int>(d_id / 2); };
    recv_domain_ids_gen<decltype(domain_to_rank)> rdig{domain_to_rank};
    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, rdig, local_domains);
    using pattern_container_type = decltype(patterns);

    auto&            d_1 = local_domains[0];
    auto&            d_2 = local_domains[1];
    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    auto func = [&ctxt](auto bi)
    {
        auto co = ghex::make_communication_object<pattern_container_type>(ctxt);
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

///** @brief Test in place receive*/
void
test_in_place_receive(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain_ipr(ctxt.rank())};
    auto&                               d = local_domains[0];

    // halo generator
    halo_generator_type hg;

    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

    // application data
    ghex::test::util::memory<int> field(d.size(), 0);
    initialize_data(d, field);
    data_descriptor_cpu_int_type data{d, field};

    // communication object
    auto co = ghex::unstructured::make_communication_object_ipr(ctxt, patterns(data));

    auto h = co.exchange();
    h.wait();

    // check exchanged data
    check_exchanged_data(d, field, patterns[0]);

#ifdef GHEX_CUDACC
    // application data
    initialize_data(d, field);
    field.clone_to_device();
    data_descriptor_gpu_int_type data_gpu{d, field.device_data(), 1, true, 0, 0};

    // communication object
    auto co_gpu = ghex::unstructured::make_communication_object_ipr(ctxt, patterns(data_gpu));

    EXPECT_NO_THROW(co_gpu.exchange());

    auto h_gpu = co_gpu.exchange();
    h_gpu.wait();

    // check exchanged data
    field.clone_to_host();
    check_exchanged_data(d, field, patterns[0]);
#endif
}

///** @brief Test in place receive with multiple fields*/
//void
//test_in_place_receive_multi(ghex::context& ctxt)
//{
//    int rank = ctxt.rank();
//
//    domain_id_type         domain_id{rank}; // 1 domain per rank
//    domain_descriptor_type d{
//        domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)};
//    std::vector<domain_descriptor_type> local_domains{d};
//    halo_generator_type                 hg{};
//
//    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);
//
//    // communication object
//    using pattern_container_type = decltype(patterns);
//    auto co = ghex::make_communication_object_ipr<pattern_container_type>(ctxt.get_communicator());
//
//    // application data
//
//    std::vector<int> field_1(d.size(), 0);
//    initialize_data(d, field_1);
//    data_descriptor_cpu_int_type data_1{d, field_1};
//
//    std::vector<int> field_2(d.size(), 0);
//    initialize_data(d, field_2);
//    data_descriptor_cpu_int_type data_2{d, field_2};
//
//    std::vector<int> field_3(d.size(), 0);
//    initialize_data(d, field_3);
//    data_descriptor_cpu_int_type data_3{d, field_3};
//
//    auto h = co.exchange(patterns(data_1), patterns(data_2), patterns(data_3));
//    h.wait();
//
//    // check exchanged data
//    check_exchanged_data(d, field_1, patterns[0]);
//    check_exchanged_data(d, field_2, patterns[0]);
//    check_exchanged_data(d, field_3, patterns[0]);
//}

///** @brief Test data descriptor concept with in-place receive*/
//void
//test_in_place_receive_oversubscribe(ghex::context& ctxt)
//{
//    int rank = ctxt.rank();
//
//    domain_id_type domain_id_1{rank * 2};
//    domain_id_type domain_id_2{
//        rank * 2 +
//        1}; // HERE domain_id, init_ordered_vertices(domain_id), init_inner_sizes(domain_id)
//    domain_descriptor_type d_1{
//        domain_id_1, init_ordered_vertices(domain_id_1), init_inner_sizes(domain_id_1)};
//    domain_descriptor_type d_2{
//        domain_id_2, init_ordered_vertices(domain_id_2), init_inner_sizes(domain_id_2)};
//    std::vector<domain_descriptor_type> local_domains{d_1, d_2};
//
//    // domain
//    std::vector<domain_descriptor_type> local_domains{make_domain_ipr(ctxt.rank()*2), make_domain_ipr(ctxt.rank()*2+1)};
//    auto& d_1 = local_domains[0];
//    auto& d_2 = local_domains[1];
//
//    // halo generator
//    halo_generator_type                 hg{};
//
//    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);
//
//    // communication object
//    using pattern_container_type = decltype(patterns);
//    auto co = ghex::make_communication_object_ipr<pattern_container_type>(ctxt.get_communicator());
//
//    // application data
//    std::vector<int> field_1(d_1.size(), 0);
//    std::vector<int> field_2(d_2.size(), 0);
//    initialize_data(d_1, field_1);
//    initialize_data(d_2, field_2);
//    data_descriptor_cpu_int_type data_1{d_1, field_1};
//    data_descriptor_cpu_int_type data_2{d_2, field_2};
//
//    EXPECT_NO_THROW(co.exchange(patterns(data_1), patterns(data_2)).wait());
//
//    auto h = co.exchange(patterns(data_1), patterns(data_2));
//    h.wait();
//
//    // check exchanged data
//    check_exchanged_data(d_1, field_1, patterns[0]);
//    check_exchanged_data(d_2, field_2, patterns[1]);
//}

/** @brief Test data descriptor concept with in-place receive and multiple threads*/
void
test_in_place_receive_threads(ghex::context& ctxt)
{
    // domain
    std::vector<domain_descriptor_type> local_domains{make_domain_ipr(ctxt.rank() * 2),
        make_domain_ipr(ctxt.rank() * 2 + 1)};
    auto&                               d_1 = local_domains[0];
    auto&                               d_2 = local_domains[1];

    // halo generator
    halo_generator_type hg;

    auto patterns = ghex::make_pattern<grid_type>(ctxt, hg, local_domains);

    std::vector<int> field_1(d_1.size(), 0);
    std::vector<int> field_2(d_2.size(), 0);
    initialize_data(d_1, field_1);
    initialize_data(d_2, field_2);
    data_descriptor_cpu_int_type data_1{d_1, field_1};
    data_descriptor_cpu_int_type data_2{d_2, field_2};

    auto func = [&ctxt](auto bi)
    {
        auto co = ghex::unstructured::make_communication_object_ipr(ctxt, bi);
        auto h = co.exchange();
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
