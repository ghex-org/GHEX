/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <gtest/gtest.h>

#include <ghex/config.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>

#include <set>
#include <vector>
#include <map>
#include <cassert>
#include <algorithm>

using domain_id_type = int;
using global_index_type = int;
using domain_descriptor_type =
    ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = ghex::unstructured::grid;
using pattern_type = ghex::pattern<grid_type::type<domain_descriptor_type>, domain_id_type>;
using local_index_type = domain_descriptor_type::local_index_type;
using local_indices_type = std::vector<local_index_type>;

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

domain_descriptor_type
make_domain(domain_id_type id)
{
    switch (id)
    {
        case 0:
        {
            std::vector<global_index_type> gids = {0, 13, 5, 2, 1, 3, 7, 11, 20};
            std::vector<local_index_type>  halo_lids = {4, 5, 6, 7, 8};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 1:
        {
            std::vector<global_index_type> gids = {1, 19, 20, 4, 7, 15, 8, 0, 9, 13, 16};
            std::vector<local_index_type>  halo_lids = {7, 8, 9, 10};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 2:
        {
            std::vector<global_index_type> gids = {3, 16, 18, 1, 5, 6};
            std::vector<local_index_type>  halo_lids = {3, 4, 5};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 3:
        {
            std::vector<global_index_type> gids = {17, 6, 11, 10, 12, 9, 0, 3, 4};
            std::vector<local_index_type>  halo_lids = {6, 7, 8};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        default:
            throw std::runtime_error("unknown domain id");
    }
}

domain_descriptor_type
make_domain_ipr(domain_id_type id)
{
    switch (id)
    {
        case 0:
        {
            std::vector<global_index_type> gids = {0, 13, 5, 2, 1, 7, 20, 3, 11};
            std::vector<local_index_type>  halo_lids = {4, 5, 6, 7, 8};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 1:
        {
            std::vector<global_index_type> gids = {1, 19, 20, 4, 7, 15, 8, 0, 13, 16, 9};
            std::vector<local_index_type>  halo_lids = {7, 8, 9, 10};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 2:
        {
            std::vector<global_index_type> gids = {3, 16, 18, 5, 1, 6};
            std::vector<local_index_type>  halo_lids = {3, 4, 5};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        case 3:
        {
            std::vector<global_index_type> gids = {17, 6, 11, 10, 12, 9, 0, 4, 3};
            std::vector<local_index_type>  halo_lids = {6, 7, 8};
            return domain_descriptor_type{id, gids.begin(), gids.end(), halo_lids.begin(),
                halo_lids.end()};
            break;
        }
        default:
            throw std::runtime_error("unknown domain id");
    }
}

void
check_domain(const domain_descriptor_type& d, const std::vector<global_index_type>& inner_gids,
    const std::vector<global_index_type>& outer_gids)
{
    EXPECT_EQ(d.inner_size(), inner_gids.size());
    EXPECT_EQ(d.size(), inner_gids.size() + outer_gids.size());
    local_index_type lid = 0;
    for (auto gid : inner_gids)
    {
        EXPECT_TRUE(d.is_inner(gid));
        EXPECT_EQ(d.inner_local_index(gid).value(), lid);
        ++lid;
    }
    for (auto gid : outer_gids)
    {
        EXPECT_TRUE(d.is_outer(gid));
        EXPECT_EQ(d.outer_ids().find(gid)->second, lid);
        ++lid;
    }
}

void
check_domain(const domain_descriptor_type& d)
{
    auto domain_id = d.domain_id();
    switch (domain_id)
    {
        case 0:
            check_domain(d, {0, 13, 5, 2}, {1, 3, 7, 11, 20});
            break;
        case 1:
            check_domain(d, {1, 19, 20, 4, 7, 15, 8}, {0, 9, 13, 16});
            break;
        case 2:
            check_domain(d, {3, 16, 18}, {1, 5, 6});
            break;
        case 3:
            check_domain(d, {17, 6, 11, 10, 12, 9}, {0, 3, 4});
            break;
    }
}

void
check_halo_generator(const domain_descriptor_type& d, const halo_generator_type& hg)
{
    auto h = hg(d);
    using vertices_set_type = std::set<global_index_type>;
    vertices_set_type halo_vertices_set;
    std::transform(h.local_indices().begin(), h.local_indices().end(),
        std::inserter(halo_vertices_set, halo_vertices_set.end()),
        [&d](auto lid) { return d.global_index(lid).value(); });
    switch (d.domain_id())
    {
        case 0:
        {
            vertices_set_type reference_halo_vertices_set{1, 20, 7, 3, 11};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 1:
        {
            vertices_set_type reference_halo_vertices_set{0, 13, 16, 9};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 2:
        {
            vertices_set_type reference_halo_vertices_set{5, 1, 6};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
        case 3:
        {
            vertices_set_type reference_halo_vertices_set{0, 4, 3};
            EXPECT_TRUE(halo_vertices_set == reference_halo_vertices_set);
            break;
        }
    }
    for (std::size_t i = 0; i < h.size(); ++i)
    {
        auto lid = h.local_indices()[i];
        auto gid = d.global_index(lid).value();
        EXPECT_TRUE(!d.is_inner(gid));
        EXPECT_EQ(d.outer_ids().find(gid)->second, lid);
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
void
check_send_halos_indices(const pattern_type& p)
{
    std::map<domain_id_type, local_indices_type> ref_map{};
    switch (p.domain_id())
    {
        case 0:
        {
            ref_map.insert({std::make_pair(domain_id_type{1}, local_indices_type{0, 1}),
                std::make_pair(domain_id_type{2}, local_indices_type{2}),
                std::make_pair(domain_id_type{3}, local_indices_type{0})});
            break;
        }
        case 1:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{0, 4, 2}),
                std::make_pair(domain_id_type{2}, local_indices_type{0}),
                std::make_pair(domain_id_type{3}, local_indices_type{3})});
            break;
        }
        case 2:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{0}),
                std::make_pair(domain_id_type{1}, local_indices_type{1}),
                std::make_pair(domain_id_type{3}, local_indices_type{0})});
            break;
        }
        case 3:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{2}),
                std::make_pair(domain_id_type{1}, local_indices_type{5}),
                std::make_pair(domain_id_type{2}, local_indices_type{1})});
            break;
        }
    }
    EXPECT_TRUE(p.send_halos().size() == 3); // size is correct
    std::set<domain_id_type> res_ids{};
    for (const auto& sh : p.send_halos())
    {
        auto res = res_ids.insert(sh.first.id);
        EXPECT_TRUE(res.second);                  // ids are unique
        EXPECT_NO_THROW(ref_map.at(sh.first.id)); // ids are correct
        EXPECT_EQ(sh.second.front().local_indices().size(),
            ref_map.at(sh.first.id).size()); // indices size is correct
        for (std::size_t idx = 0; idx < ref_map.at(sh.first.id).size(); ++idx)
        {
            EXPECT_EQ(sh.second.front().local_indices()[idx],
                ref_map.at(sh.first.id)[idx]); // indices are correct
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
void
check_recv_halos_indices(const pattern_type& p)
{
    std::map<domain_id_type, local_indices_type> ref_map{};
    switch (p.domain_id())
    {
        case 0:
        {
            ref_map.insert({std::make_pair(domain_id_type{1}, local_indices_type{4, 6, 8}),
                std::make_pair(domain_id_type{2}, local_indices_type{5}),
                std::make_pair(domain_id_type{3}, local_indices_type{7})});
            break;
        }
        case 1:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{7, 9}),
                std::make_pair(domain_id_type{2}, local_indices_type{10}),
                std::make_pair(domain_id_type{3}, local_indices_type{8})});
            break;
        }
        case 2:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{4}),
                std::make_pair(domain_id_type{1}, local_indices_type{3}),
                std::make_pair(domain_id_type{3}, local_indices_type{5})});
            break;
        }
        case 3:
        {
            ref_map.insert({std::make_pair(domain_id_type{0}, local_indices_type{6}),
                std::make_pair(domain_id_type{1}, local_indices_type{8}),
                std::make_pair(domain_id_type{2}, local_indices_type{7})});
            break;
        }
    }
    EXPECT_EQ(p.recv_halos().size(), 3u); // size is correct
    std::set<domain_id_type> res_ids{};
    for (const auto& rh : p.recv_halos())
    {
        auto res = res_ids.insert(rh.first.id);
        EXPECT_TRUE(res.second);                  // ids are unique
        EXPECT_NO_THROW(ref_map.at(rh.first.id)); // ids are correct
        EXPECT_EQ(rh.second.front().local_indices().size(),
            ref_map.at(rh.first.id).size()); // indices size is correct
        for (std::size_t idx = 0; idx < ref_map.at(rh.first.id).size(); ++idx)
        {
            EXPECT_EQ(rh.second.front().local_indices()[idx],
                ref_map.at(rh.first.id)[idx]); // indices are correct
        }
    }
}

template<typename Container>
void
initialize_data(const domain_descriptor_type& d, Container& field, std::size_t levels = 1u, bool levels_first = true)
{
    assert(field.size() == d.size() * levels);
    if (levels_first)
        for (const auto& x : d.inner_ids())
            for (std::size_t level = 0u; level < levels; ++level)
                field[x.second * levels + level] = d.domain_id() * 10000 + x.first*100 + level;
    else
        for (std::size_t level = 0u; level < levels; ++level)
            for (const auto& x : d.inner_ids())
                field[x.second + level*d.size()] = d.domain_id() * 10000 + x.first*100 + level;
}

template<typename Container>
void
check_exchanged_data(const domain_descriptor_type& d, const Container& field, const pattern_type& p, std::size_t levels = 1u, bool levels_first = true)
{
    using value_type = typename Container::value_type;
    using index_type = pattern_type::index_type;
    std::map<index_type, domain_id_type> halo_map{};
    for (const auto& [edid, c]: p.recv_halos())
    {
        for (const auto idx : c.front().local_indices())
        {
            halo_map.insert(std::make_pair(idx, edid.id));
        }
    }
    if (levels_first)
        for (auto [idx, did] : halo_map)
            for (std::size_t level = 0u; level < levels; ++level)
                EXPECT_EQ(field[idx * levels + level], static_cast<value_type>(did * 10000 + d.global_index(idx).value()*100 + level));
    else
        for (std::size_t level = 0u; level < levels; ++level)
            for (auto [idx, did] : halo_map)
                EXPECT_EQ(field[idx + level * d.size()], static_cast<value_type>(did * 10000 + d.global_index(idx).value()*100 + level));
}

/** @brief Helper functor type, used as default template argument below*/
struct domain_to_rank_identity
{
    int operator()(const domain_id_type d_id) const { return static_cast<int>(d_id); }
};

/** @brief Ad hoc receive domain ids generator, valid only for this specific test case.
 * Even if the concept is general, the implementation of the operator() is appplication-specific.*/
template<typename DomainToRankFunc = domain_to_rank_identity>
class recv_domain_ids_gen
{
  public:
    class halo
    {
      private:
        std::vector<domain_id_type> m_domain_ids;
        local_indices_type          m_remote_indices;
        std::vector<int>            m_ranks;

      public:
        halo() noexcept = default;
        halo(const std::vector<domain_id_type>& domain_ids,
            const local_indices_type& remote_indices, const std::vector<int>& ranks)
        : m_domain_ids{domain_ids}
        , m_remote_indices{remote_indices}
        , m_ranks{ranks}
        {
        }
        const std::vector<domain_id_type>& domain_ids() const noexcept { return m_domain_ids; }
        const local_indices_type& remote_indices() const noexcept { return m_remote_indices; }
        const std::vector<int>&   ranks() const noexcept { return m_ranks; }
    };

  private:
    const DomainToRankFunc& m_func;

  public:
    recv_domain_ids_gen(const DomainToRankFunc& func = domain_to_rank_identity{})
    : m_func{func}
    {
    }

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
    halo operator()(const domain_descriptor_type& domain) const
    {
        std::vector<domain_id_type> domain_ids{};
        local_indices_type          remote_indices{};
        switch (domain.domain_id())
        {
            case 0:
            {
                domain_ids.insert(domain_ids.end(), {1, 2, 1, 3, 1});
                remote_indices.insert(remote_indices.end(), {0, 0, 4, 2, 2});
                break;
            }
            case 1:
            {
                domain_ids.insert(domain_ids.end(), {0, 3, 0, 2});
                remote_indices.insert(remote_indices.end(), {0, 5, 1, 1});
                break;
            }
            case 2:
            {
                domain_ids.insert(domain_ids.end(), {1, 0, 3});
                remote_indices.insert(remote_indices.end(), {0, 2, 1});
                break;
            }
            case 3:
            {
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
