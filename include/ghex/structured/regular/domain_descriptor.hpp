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

#include <type_traits>
#include <array>
#include <algorithm>

namespace ghex
{
namespace structured
{
namespace regular
{
/** @brief implements domain descriptor concept for structured domains
  * @tparam DomainIdType domain id type
  * @tparam Dimension dimension of domain*/
template<typename DomainIdType, typename Dimension_t>
class domain_descriptor;

template<typename DomainIdType, int Dimension>
class domain_descriptor<DomainIdType, std::integral_constant<int, Dimension>>
{
  public: // member types
    using domain_id_type = DomainIdType;
    using dimension = std::integral_constant<int, Dimension>;
    using coordinate_type = std::array<int, dimension::value>;

  public: // ctors
    /** @brief construct a local domain
      * @tparam Array coordinate-like type
      * @param id domain id
      * @param first for each dimension the first coordinate in domain (global coordinate)
      * @param last for each dimension the last coordinate in domain (including, global coordinate) */
    template<typename Array>
    domain_descriptor(domain_id_type id, const Array& first, const Array& last)
    : m_id{id}
    {
        std::copy(std::begin(first), std::end(first), m_first.begin());
        std::copy(std::begin(last), std::end(last), m_last.begin());
    }

  public: // member functions
    domain_id_type         domain_id() const { return m_id; }
    const coordinate_type& first() const { return m_first; }
    const coordinate_type& last() const { return m_last; }

  private: // members
    domain_id_type  m_id;
    coordinate_type m_first;
    coordinate_type m_last;
};

} // namespace regular
} // namespace structured
} // namespace ghex
