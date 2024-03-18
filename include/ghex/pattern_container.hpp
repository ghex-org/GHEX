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

#include <ghex/context.hpp>
#include <ghex/buffer_info.hpp>
#include <vector>

namespace ghex
{
namespace detail
{
// forward declaration
template<typename GridType>
struct make_pattern_impl;
} // namespace detail

// forward declaration
template<typename GridType, typename DomainIdType>
class pattern;

/** @brief an iterable holding communication patterns (one pattern per domain)
  * @tparam GridType indicates structured/unstructured grids
  * @tparam DomainIdType type to uniquely identify partail (local) domains*/
template<typename GridType, typename DomainIdType>
class pattern_container
{
  public: // member tyes
    using communicator_type = context::communicator_type;
    using grid_type = GridType;
    using domain_id_type = DomainIdType;
    /** @brief pattern type this object is holding */
    using value_type = pattern<GridType, DomainIdType>;

  private: // private member types
    using data_type = std::vector<value_type>;

  private: // friend declarations
    friend struct detail::make_pattern_impl<GridType>;

  public: // copy constructor
    pattern_container(const pattern_container& other)
    : m_patterns{other.m_patterns}
    , m_max_tag{other.m_max_tag}
    {
        for (auto& p : m_patterns) p.m_container = this;
    }

    pattern_container(pattern_container&& other)
    : m_patterns{std::move(other.m_patterns)}
    , m_max_tag{other.m_max_tag}
    {
        for (auto& p : m_patterns) p.m_container = this;
    }

    //private: // private constructor called through make_pattern
    pattern_container(context& c, data_type&& d, int mt) noexcept
    : m_ctxt{c.transport_context()}
    , m_patterns(d)
    , m_max_tag(mt)
    {
        for (auto& p : m_patterns) p.m_container = this;
    }

  public: // member functions
    int         size() const noexcept { return m_patterns.size(); }
    const auto& operator[](int i) const noexcept { return m_patterns[i]; }
    auto&       operator[](int i) noexcept { return m_patterns[i]; }
    auto        begin() noexcept { return m_patterns.begin(); }
    auto        end() noexcept { return m_patterns.end(); }
    auto        begin() const noexcept { return m_patterns.cbegin(); }
    auto        end() const noexcept { return m_patterns.cend(); }
    auto        cbegin() const noexcept { return m_patterns.cbegin(); }
    auto        cend() const noexcept { return m_patterns.cend(); }
    int         max_tag() const noexcept { return m_max_tag; }

    /** @brief bind a field to a pattern
      * @tparam Field field type
      * @param field field instance
      * @return lightweight buffer_info object. Attention: holds references to field and pattern! */
    template<typename Field>
    buffer_info<value_type, typename Field::arch_type, Field> operator()(Field& field) const
    {
        // linear search here
        for (auto& p : m_patterns)
            if (p.domain_id() == field.domain_id()) return p(field);
        throw std::runtime_error("field incompatible with available domains!");
    }

  private: // members
    oomph::context* m_ctxt;
    data_type m_patterns;
    int       m_max_tag;
};

/** @brief construct a pattern for each domain and establish neighbor relationships
  * @tparam GridType indicates structured/unstructured grids
  * @tparam HaloGenerator function object which takes a domain as argument
  * @tparam DomainRange a range type holding domains
  * @param context transport layer context
  * @param hgen receive halo generator function object (emits iteration spaces (global
  * coordinates) or index lists (global indices)
  * @param d_range range of local domains
  * @return iterable of patterns (one per domain) */
template<typename GridType, typename HaloGenerator, typename DomainRange>
auto
make_pattern(context& c, HaloGenerator&& hgen, DomainRange&& d_range)
{
    using grid_type =
        typename GridType::template type<typename std::remove_reference_t<DomainRange>::value_type>;
    return detail::make_pattern_impl<grid_type>::apply(
        c, std::forward<HaloGenerator>(hgen), std::forward<DomainRange>(d_range));
}

/** @brief construct a pattern for each domain and establish neighbor relationships, with
 * user-defined function for recv domain ids.
 * TO DO: so far, the structured specialization just redirects to the previous one
 * (recv_domains_gen is just ignored)
 * @tparam GridType indicates structured/unstructured grids
 * @tparam HaloGenerator function object which takes a domain as argument
 * @tparam RecvDomainsGenerator function object which takes a domain as argument
 * @tparam DomainRange a range type holding domains
 * @param context transport layer context
 * @param hgen receive halo generator function object (emits iteration spaces (global 
 * coordinates) or index lists (global indices)
 * @param recv_domains_gen function object which emits recv domain ids (one for each 
 * iteration space / index list, as provided by hgen)
 * @param d_range range of local domains
 * @return iterable of patterns (one per domain) */
template<typename GridType, typename HaloGenerator, typename RecvDomainIdsGen, typename DomainRange>
auto
make_pattern(
    context& c, HaloGenerator&& hgen, RecvDomainIdsGen&& recv_domain_ids_gen, DomainRange&& d_range)
{
    using grid_type =
        typename GridType::template type<typename std::remove_reference_t<DomainRange>::value_type>;
    return detail::make_pattern_impl<grid_type>::apply(c, std::forward<HaloGenerator>(hgen),
        std::forward<RecvDomainIdsGen>(recv_domain_ids_gen), std::forward<DomainRange>(d_range));
}

} // namespace ghex
