/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <vector>

#include <fhex/ghex_defs.hpp>
#include <fhex/obj_wrapper.hpp>
#include <fhex/context_bind.hpp>

#include <ghex/unstructured/user_concepts.hpp> // ghex arch traits included here
#include <ghex/unstructured/pattern.hpp>       // grid and pattern_container included here
#include <ghex/communication_object.hpp>

namespace fhex
{
using unstruct_domain_descriptor_type = ghex::unstructured::domain_descriptor<int, int>;
using unstruct_halo_generator_type = ghex::unstructured::halo_generator<int, int>;
using unstruct_field_descriptor_cpu_type =
    ghex::unstructured::data_descriptor<ghex::cpu, int, int, fp_type>;
using unstruct_grid_type = ghex::unstructured::grid;
using unstruct_grid_detail_type = ghex::unstructured::detail::grid<std::size_t>;
using unstruct_pattern_type = ghex::pattern<unstruct_grid_detail_type, int>;
using unstruct_pattern_container_type = ghex::pattern_container<unstruct_grid_detail_type, int>;
using unstruct_buffer_info_type =
    ghex::buffer_info<unstruct_pattern_type, ghex::cpu, unstruct_field_descriptor_cpu_type>;
using unstruct_communication_object_type =
    ghex::communication_object<unstruct_grid_detail_type, int>;
using unstruct_exchange_handle_type = unstruct_communication_object_type::handle_type;

/** @brief wraps domain descriptor constructor arguments
 * When applied to the ICON use case,
 * these parameters have to be inferred from the t-patch.
 * In particular, vertices should match the field with domain global indices;
 * total_size is the total domain size, including halo indices;
 * finally, inner_size is the actual domain size, without halo indices.
 * inner_size has to be computed based on a second field available in ICON,
 * which uses different flags for inner and halo indices,
 * and therefore allows to identify and count inner indices.
*/
struct ghex_unstruct_domain_desc
{
    int  id;
    int* vertices = nullptr;
    int  total_size;
    int* outer_indices = nullptr;
    int  outer_size;
};

struct ghex_unstruct_field_desc
{
    int      domain_id;
    int      domain_size;
    int      levels;
    fp_type* field = nullptr;
};

/** @brief wraps a container of buffer info objects
 * This is the only class which does not have a 1 to 1 match in GHEX.
 * It has been introduced to overcome the issue with
 * binding the the variadic template based signature of the exchange method.
 * An exchange_args type allows for binding to the iterator based one instead.
 * Note that the actual GHEX field_descriptor object
 * is created when calling the add member function,
 * since the ghex_unstruct_field_descriptor f, passed as argument,
 * is only a wrapper aronud the field_descriptor constructor arguments.
*/
class exchange_args
{
  public: // member types
    using arg_type = unstruct_buffer_info_type;

  public: // ctors
    exchange_args() noexcept
    : m_patterns{}
    , m_fields{}
    , m_args{}
    , m_valid{true}
    {
    }

    exchange_args(const exchange_args&) noexcept = delete;
    exchange_args(exchange_args&&) noexcept = default; // TO DO: m_valid?

    exchange_args& operator=(const exchange_args&) noexcept = delete;
    exchange_args& operator=(exchange_args&&) noexcept = default; // TO DO: m_valid?

  public:                                                                     // member functions
    void add(unstruct_pattern_container_type* p, ghex_unstruct_field_desc* f) // TO DO: const ref?
    {
        m_patterns.push_back(p); // TO DO: emplace_back?
        if (m_fields.size() == m_fields.capacity()) m_valid = false;
        m_fields.emplace_back(f->domain_id, f->domain_size, f->field, f->levels);
        if (m_valid) { m_args.emplace_back(m_patterns.back()->operator()(m_fields.back())); }
        else { validate(); }
    }

    auto begin() { return m_args.begin(); }
    auto end() { return m_args.end(); }

    const std::vector<arg_type>& get() const { return m_args; }
    std::vector<arg_type>&       get() { return m_args; }

  private: // member functions
    void validate()
    {
        if (!m_valid)
        {
            m_args.clear();
            for (std::size_t i = 0; i < m_patterns.size(); ++i)
            {
                m_args.emplace_back(m_patterns[i]->operator()(m_fields[i]));
            }
            m_valid = true;
        }
    }

  private:
    std::vector<unstruct_pattern_container_type*>   m_patterns;
    std::vector<unstruct_field_descriptor_cpu_type> m_fields;
    std::vector<arg_type>                           m_args;
    bool                                            m_valid;
};

extern "C" void
ghex_unstruct_pattern_setup_impl(obj_wrapper** pattern, ghex_unstruct_domain_desc* domain_descs,
    int n_domains) // TO DO: check wrapper (everywhere obj_wrapper occurs)
{
    std::vector<unstruct_domain_descriptor_type> local_domains{};
    local_domains.reserve(n_domains);
    for (int i = 0; i < n_domains; ++i)
    {
        const auto& d = domain_descs[i];
        local_domains.emplace_back(d.id, d.vertices, d.vertices + d.total_size, d.outer_indices,
            d.outer_indices + d.outer_size);
    }
    unstruct_halo_generator_type hg{};
    *pattern =
        new obj_wrapper{ghex::make_pattern<unstruct_grid_type>(context(), hg, local_domains)};
}

extern "C" void
ghex_unstruct_communication_object_init(obj_wrapper** co)
{
    *co = new obj_wrapper{
        ghex::make_communication_object<unstruct_pattern_container_type>(context())};
}

extern "C" void
ghex_unstruct_exchange_args_init(obj_wrapper** args)
{
    *args = new obj_wrapper{exchange_args{}};
}

extern "C" void
ghex_unstruct_exchange_args_add(obj_wrapper** args, obj_wrapper** pattern,
    ghex_unstruct_field_desc*
        field_desc) // TO DO: check whether explicitly pass pointers from Fortran
{
    exchange_args*                   args_ptr = get_object_ptr_unsafe<exchange_args>(*args);
    unstruct_pattern_container_type* pattern_ptr =
        get_object_ptr_unsafe<unstruct_pattern_container_type>(*pattern);
    args_ptr->add(pattern_ptr, field_desc);
}

extern "C" void*
ghex_unstruct_exchange(obj_wrapper** co, obj_wrapper** args)
{
    unstruct_communication_object_type* co_ptr =
        get_object_ptr_unsafe<unstruct_communication_object_type>(*co);
    exchange_args* args_ptr = get_object_ptr_unsafe<exchange_args>(*args);
    return new obj_wrapper{co_ptr->exchange(args_ptr->begin(), args_ptr->end())};
}

extern "C" void
ghex_unstruct_exchange_handle_wait(obj_wrapper** h)
{
    unstruct_exchange_handle_type* h_ptr = get_object_ptr_unsafe<unstruct_exchange_handle_type>(*h);
    h_ptr->wait();
}
} // namespace fhex
