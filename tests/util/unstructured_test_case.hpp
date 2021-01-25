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
#ifndef TESTS_UTIL_UNSTRUCTURED_TEST_CASE_HPP
#define TESTS_UTIL_UNSTRUCTURED_TEST_CASE_HPP

#include <set>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>

#include <gtest/gtest.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/arch_list.hpp>


#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
#else
using transport = gridtools::ghex::tl::ucx_tag;
#endif
using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
using communicator_type = context_type::communicator_type;
using domain_id_type = int;
using global_index_type = int;
using domain_descriptor_type = gridtools::ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = gridtools::ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = gridtools::ghex::unstructured::grid;
using pattern_type = gridtools::ghex::pattern<communicator_type, grid_type::type<domain_descriptor_type>, domain_id_type>;
using vertices_type = domain_descriptor_type::vertices_type;
using vertices_set_type = domain_descriptor_type::vertices_set_type;
using adjncy_type = domain_descriptor_type::adjncy_type;
using map_type = domain_descriptor_type::map_type;
using local_index_type = domain_descriptor_type::local_index_type;
using local_indices_type = std::vector<local_index_type>;
using it_diff_type = vertices_type::iterator::difference_type;


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
    for (std::size_t i = 0; i < h.size(); ++i) {
        EXPECT_TRUE(d.vertices()[h.local_indices()[i]] == h.vertices()[i]);
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
        EXPECT_TRUE(sh.second.front().local_indices().size() == ref_map.at(sh.first.id).size()); // indices size is correct
        for (std::size_t idx = 0; idx < ref_map.at(sh.first.id).size(); ++idx) {
            EXPECT_TRUE(sh.second.front().local_indices()[idx] == ref_map.at(sh.first.id)[idx]); // indices are correct
        }
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
        EXPECT_TRUE(rh.second.front().local_indices().size() == ref_map.at(rh.first.id).size()); // indices size is correct
        for (std::size_t idx = 0; idx < ref_map.at(rh.first.id).size(); ++idx) {
            EXPECT_TRUE(rh.second.front().local_indices()[idx] == ref_map.at(rh.first.id)[idx]); // indices are correct
        }
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


/** @brief Helper functor type, used as default template argument below*/
struct domain_to_rank_identity {
    int operator()(const domain_id_type d_id) const {
        return static_cast<int>(d_id);
    }
};


/** @brief Ad hoc receive domain ids generator, valid only for this specific test case.
 * Even if the concept is general, the implementation of the operator() is appplication-specific.*/
template <typename DomainToRankFunc = domain_to_rank_identity>
class recv_domain_ids_gen {

    public:

        class halo {

            private:

                std::vector<domain_id_type> m_domain_ids;
                local_indices_type m_remote_indices;
                std::vector<int> m_ranks;

            public:

                halo() noexcept = default;
                halo(const std::vector<domain_id_type>& domain_ids,
                     const local_indices_type& remote_indices,
                     const std::vector<int>& ranks) :
                    m_domain_ids{domain_ids},
                    m_remote_indices{remote_indices},
                    m_ranks{ranks} {}
                const std::vector<domain_id_type>& domain_ids() const noexcept { return m_domain_ids; }
                const local_indices_type& remote_indices() const noexcept { return m_remote_indices; }
                const std::vector<int>& ranks() const noexcept { return m_ranks; }

        };

    private:

        const DomainToRankFunc& m_func;

    public:

        recv_domain_ids_gen(const DomainToRankFunc& func = domain_to_rank_identity{}) : m_func{func} {}

        // member functions (operator ())
        /* Domains
         *
         *             id  |          inner           |        halo        |  recv_domain_ids  |  remote_indices  |
         *             --------------------------------------------------------------------------------------------
         *              0  | [0, 13, 5, 2]            | [1, 3, 7, 11, 20]  | [1, 2, 1, 3, 1]   | [0, 0, 4, 2, 2]  |
         *              1  | [1, 19, 20, 4, 7, 15, 8] | [0, 9, 13, 16]     | [0, 3, 0, 2]      | [0, 5, 1, 1]     |
         *              2  | [3, 16, 18]              | [1, 5, 6]          | [1, 0, 3]         | [0, 2, 1]        |
         *              3  | [17, 6, 11, 10, 12, 9]   | [0, 3, 4]          | [0, 2, 1]         | [0, 0, 3]        |
         *
         * */
        halo operator()(const domain_descriptor_type& domain) const {
            std::vector<domain_id_type> domain_ids{};
            local_indices_type remote_indices{};
            switch (domain.domain_id()) {
                case 0: {
                    domain_ids.insert(domain_ids.end(), {1, 2, 1, 3, 1});
                    remote_indices.insert(remote_indices.end(), {0, 0, 4, 2, 2});
                    break;
                }
                case 1: {
                    domain_ids.insert(domain_ids.end(), {0, 3, 0, 2});
                    remote_indices.insert(remote_indices.end(), {0, 5, 1, 1});
                    break;
                }
                case 2: {
                    domain_ids.insert(domain_ids.end(), {1, 0, 3});
                    remote_indices.insert(remote_indices.end(), {0, 2, 1});
                    break;
                }
                case 3: {
                    domain_ids.insert(domain_ids.end(), {0, 2, 1});
                    remote_indices.insert(remote_indices.end(), {0, 0, 3});
                    break;
                }
            }
            std::vector<int> ranks(domain_ids.size());
            std::transform(domain_ids.begin(), domain_ids.end(), ranks.begin(), m_func);
            return {domain_ids, remote_indices, ranks};
        }

};


/** @brief Vertices generator to be used for in-place receive tests.
 * Inner vertices are the same as before, halo vertices are partitioned according to recv domain id.*/
vertices_type init_ordered_vertices(const domain_id_type domain_id) {
    switch (domain_id) {
        case 0: {
            return {0, 13, 5, 2, 1, 7, 20, 3, 11};
        }
        case 1: {
            return {1, 19, 20, 4, 7, 15, 8, 0, 13, 16, 9};
        }
        case 2: {
            return {3, 16, 18, 5, 1, 6};
        }
        case 3: {
            return {17, 6, 11, 10, 12, 9, 0, 4, 3};
        }
        default: {
            return {};
        }
    }
}


/** @brief Simple generator of domain inner sizes, used for in-place receive tests.*/
std::size_t init_inner_sizes(const domain_id_type domain_id) {
    switch (domain_id) {
        case 0: return 4;
        case 1: return 7;
        case 2: return 3;
        case 3: return 6;
        default: return 0;
    }
}

#endif /* TESTS_UTIL_UNSTRUCTURED_TEST_CASE_HPP */
