/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <fhex/obj_wrapper.hpp>
#include <fhex/context_bind.hpp>

#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/unstructured/pattern.hpp> // grid and pattern_container included here

namespace fhex
{
using unstruct_domain_descriptor_type = ghex::unstructured::domain_descriptor<int, int>;
using unstruct_halo_generator_type = ghex::unstructured::halo_generator<int, int>;
using unstruct_grid_type = ghex::unstructured::grid;
using unstruct_grid_detail_type = ghex::unstructured::detail::grid<std::size_t>;
using unstruct_pattern_container_type = ghex::pattern_container<unstruct_grid_detail_type, int>;

struct ghex_unstruct_domain_desc
{
    int id;
    int* vertices = nullptr;
    int total_size;
    int inner_size;
    int levels;
};

struct ghex_unstruct_field_desc
{
    int domain_id;
    int domain_size;
    int levels;
    fp_type* field = nullptr;
};

extern "C" void
ghex_unstruct_pattern_setup_impl(obj_wrapper** pattern, ghex_unstruct_domain_desc* domain_descs, int n_domains) // TO DO: check wrapper
{
    std::vector<unstruct_domain_descriptor_type> local_domains{};
    local_domain.reserve(n_domains);
    for (std::size_t i = 0; i < n_domains; ++i)
    {
        const auto& d = domain_descs[i];
        local_domains.emplace_back(d.id, d.vertices, d.total_size, d.inner_size, d.levels);
    }
    unstruct_halo_generator_type hg{};
    *pattern = new obj_wrapper{ghex::make_pattern<unstruct_grid_type>(context(), hg, local_domains)};
}

extern "C" void
ghex_unstruct_communication_object_init(obj_wrapper** co) // TO DO: check wrapper
{
    *co = new obj_wrapper{ghex::make_communication_object<unstruct_pattern_container_type>(context())};
}
} // namespace fhex
