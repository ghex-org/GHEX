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

#include <tuple>
#include <ghex/util/distribution.hpp>

namespace ghex
{
namespace detail
{
template<typename IndexSequence>
struct dist_1D_generator;

template<std::size_t... Is>
struct dist_1D_generator<std::index_sequence<Is...>>
{
    using dist_1D_tuple_type = std::tuple<hierarchical_distribution<sizeof...(Is) - Is>...>;

    template<typename Dims, std::size_t... Js>
    static hierarchical_distribution<sizeof...(Js)> generate_1(
        const Dims& dims, std::index_sequence<Js...>) noexcept
    {
        return {{dims[sizeof...(Is) - sizeof...(Js) + 1 + Js].size()...}, false};
    }

    template<typename Dims>
    static dist_1D_tuple_type generate(const Dims& dims) noexcept
    {
        return {generate_1(dims, std::make_index_sequence<sizeof...(Is) - Is>())...};
    }
};

} // namespace detail

/** @brief A class that stores a hierarchical resource layout. The hierarchy is comprised of
  * D-dimensional hypercubes, where each hypercube holds next level's hypercube at each coordinate.
  * @tparam D dimension
  * @tparam Levels number of levels*/
template<unsigned int D, unsigned int Levels>
class hierarchical_resource_layout
{
    static_assert(D > 0, "dimension must be greater than 0");
    static_assert(Levels > 1, "number of levels must be greater than 1");

  private:
    using distribution_type = hierarchical_distribution<Levels>;
    using dims_map_type = dims_map<D>;
    using dims_map_array_type = std::array<dims_map_type, Levels>;
    using dist_1D_generator = detail::dist_1D_generator<std::make_index_sequence<Levels - 1>>;
    using dist_1D_tuple_type = typename dist_1D_generator::dist_1D_tuple_type;

  public:
    using size_type = typename dims_map_type::size_type;
    using array_type = typename dims_map_type::array_type;

  private:
    dims_map_array_type m_dims;
    distribution_type   m_dist;
    dist_1D_tuple_type  m_1D_dist;

  private:
    template<std::size_t... Is>
    static distribution_type make_dist(
        const dims_map_array_type& d, std::index_sequence<Is...>) noexcept
    {
        return {{d[Is].size()...}, true};
    }

  public:
    /** @brief Construct from extents
      * @tparam Arrays array convertible to array_type
      * @param dims D-dimensional extent arrays for each level */
    template<typename... Arrays>
    hierarchical_resource_layout(const Arrays&... dims)
    : m_dims{dims_map_type(dims, false)...}
    , m_dist(make_dist(m_dims, std::make_index_sequence<Levels>()))
    , m_1D_dist{dist_1D_generator::generate(m_dims)}
    {
    }

    hierarchical_resource_layout(const hierarchical_resource_layout&) = default;
    hierarchical_resource_layout& operator=(const hierarchical_resource_layout&) = default;

  public:
    /** returns total number of resources */
    size_type size() const noexcept { return m_dist.size(); }

    /** returns number of resources on level I per parent level I-1 */
    template<unsigned int I>
    size_type relative_size() const noexcept
    {
        static_assert(I < Levels, "level index I must be within the range of specified levels");
        return m_dims[I].size();
    }

    /** returns level I index, given a global resource index */
    template<unsigned int I>
    size_type index(size_type idx) const noexcept
    {
        static_assert(I < Levels, "level index I must be within the range of specified levels");
        return m_dist(idx)[I];
    }

    /** returns resource index relative to level I, given a global resource index */
    template<unsigned int I>
    size_type relative_resource(size_type idx) const noexcept
    {
        static_assert(I < (Levels - 1), "level I must be lower than greatest level");
        return relative_resource(idx, std::integral_constant<unsigned int, I>());
    }

    /** returns last coordinate in global domain */
    array_type last_coord() const noexcept { return this->operator()(size() - 1); }

    /** returns spatial coordinate given a resource index */
    array_type operator()(size_type idx) const noexcept
    {
        array_type res;
        for (unsigned i = 0; i < D; ++i) res[i] = 0;
        const auto indices = m_dist(idx);
        get_coord(indices, res, std::integral_constant<unsigned int, Levels - 1>());
        return res;
    }

  private:
    size_type relative_resource(
        size_type idx, std::integral_constant<unsigned int, 0>) const noexcept
    {
        return idx - index<0>(idx) * std::get<0>(m_1D_dist).size();
    }

    template<unsigned int I>
    size_type relative_resource(
        size_type idx, std::integral_constant<unsigned int, I>) const noexcept
    {
        return relative_resource(idx, std::integral_constant<unsigned int, I - 1>()) -
               index<I>(idx) * std::get<I>(m_1D_dist).size();
    }

    template<typename Array>
    void get_coord(const Array& indices, array_type& coord,
        std::integral_constant<unsigned int, 0>) const noexcept
    {
        constexpr unsigned int J = Levels - 1;
        const auto             level_coord = m_dims[J](indices[J]);
        for (unsigned i = 0; i < D; ++i) coord[i] = level_coord[i] + coord[i] * m_dims[J].dims()[i];
    }

    template<typename Array, unsigned int I>
    void get_coord(const Array& indices, array_type& coord,
        std::integral_constant<unsigned int, I>) const noexcept
    {
        constexpr unsigned int J = Levels - I - 1;
        const auto             level_coord = m_dims[J](indices[J]);
        for (unsigned i = 0; i < D; ++i) coord[i] = level_coord[i] + coord[i] * m_dims[J].dims()[i];
        get_coord(indices, coord, std::integral_constant<unsigned int, I - 1>());
    }
};

} // namespace ghex
