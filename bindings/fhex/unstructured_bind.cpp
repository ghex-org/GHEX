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

#include <ghex/unstructured/user_concepts.hpp> // ghex arch traits included here
#include <ghex/unstructured/pattern.hpp> // grid and pattern_container included here

namespace fhex
{
using unstruct_domain_descriptor_type = ghex::unstructured::domain_descriptor<int, int>;
using unstruct_halo_generator_type = ghex::unstructured::halo_generator<int, int>;
using unstruct_field_descriptor_cpu_type = ghex::unstructured::data_descriptor<ghex::cpu, int, int, fp_type>;
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

class exchange_args
{
  public: // member types
    using arg_type = std::pair<unstruct_pattern_container_type*, unstruct_field_descriptor_cpu_type>; // TO DO: const ref?

  public: // ctors
    exchange_args() noexcept = default;

    exchange_args(const exchange_args&) noexcept = default;
    exchange_args(exchange_args&&) noexcept = default;

    exchange_args& operator=(const exchange_args&) noexcept = default;
    exchange_args& operator=(exchange_args&&) noexcept = default;

  public: // member functions
    void add(unstruct_pattern_container_type* p, ghex_unstruct_field_desc* f) // TO DO: const ref?
    {
        m_args.emplace_back({p, {f->domain_id, f->domain_size, f->levels, f->field}}); // TO DO: check constructors used and std::forward
    }

    const std::vector<arg_type>& get_args() const
    {
        return m_args;
    }
    std::vector<arg_type>& get_args()
    {
        return m_args;
    }

  private:
    std::vector<arg_type> m_args;
};

extern "C" void
ghex_unstruct_pattern_setup_impl(obj_wrapper** pattern, ghex_unstruct_domain_desc* domain_descs, int n_domains) // TO DO: check wrapper (everywhere obj_wrapper occurs)
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
ghex_unstruct_communication_object_init(obj_wrapper** co)
{
    *co = new obj_wrapper{ghex::make_communication_object<unstruct_pattern_container_type>(context())};
}

extern "C" void
ghex_unstruct_exchange_args_init(obj_wrapper** args)
{
    *args = new obj_wrapper{exchange_args{}};
}

extern "C" void
ghex_unstruct_exchange_args_add(obj_wrapper** args, obj_wrapper** pattern, ghex_unstruct_field_desc* field_desc) // TO DO: check whether explicitly pass pointers from Fortran
{
    exchange_args* args_ptr = get_object_ptr_unsafe<exchange_args>(*args);
    unstruct_pattern_container_type* pattern_ptr = get_object_ptr_unsafe<unstruct_pattern_container_type>(*pattern);
    args_ptr->add(pattern_ptr, field_desc);
}
} // namespace fhex
