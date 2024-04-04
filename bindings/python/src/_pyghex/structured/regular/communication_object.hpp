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
#include <ghex/communication_object.hpp>
#include <structured/types.hpp>

#include <iterator>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{

using communication_object_args =
    gridtools::meta::cartesian_product<types::grids, types::domain_ids>;

using communication_object_specializations =
    gridtools::meta::transform<gridtools::meta::rename<ghex::communication_object>::template apply,
        communication_object_args>;
} // namespace


// Communication object specializations are stored in a variant and constructed on demand before the first exchange.
// - this removes the need to inject the pattern type at construction, i.e.
//   in the python function `make_communication_object` doesn't require a pattern object to infer the type anymore
// - if this communication object shim is later used with a different *type* of pattern, for example
//   a 2d pattern instead of a 3d pattern, the exchange will fail with an exception
struct communication_object_shim {
    // the variant's first alternative is of type std::monostate to indicate the empty state
    using variant_t =
        gridtools::meta::rename<std::variant,
            gridtools::meta::push_front<communication_object_specializations, std::monostate>>;
    ghex::context* ctx = nullptr;
    variant_t m;

    // exchange of buffer info objects
    template<typename... Patterns, typename... Archs, typename... Fields>
    auto exchange(ghex::buffer_info<Patterns, Archs, Fields>&... b) {
        return get_co<gridtools::meta::list<Patterns...>>().exchange(b...);
    }

    // exchange of iterator pairs pointing to buffer info ranges
    template<typename... Its>
    auto exchange(Its... its) {
        // need even number of iterators (begin and end)
        static_assert(sizeof...(Its) % 2 == 0);
        return exchange_from_iterators(std::make_tuple(std::move(its)...), std::make_index_sequence<sizeof...(Its)/2>());
    }

  private:
    // extractors for nested typedefs
    template<typename It>
    using get_pattern_t = typename std::iterator_traits<It>::value_type::pattern_type;
    template<typename P>
    using get_grid = typename P::grid_type;
    template<typename P>
    using get_did = typename P::domain_id_type;

    // helper function for iterators
    template<typename... Its, std::size_t... Is>
    auto exchange_from_iterators(std::tuple<Its...> t, std::index_sequence<Is...>) {
        // every second iterator is a begin
        using begins = decltype(std::make_tuple(std::get<Is*2>(t)...));
        static constexpr std::size_t half_size = sizeof...(Is);
        return get_co<gridtools::meta::transform<get_pattern_t, begins>>().exchange(
            std::get<Is>(t)..., std::get<Is + half_size>(t)...);
    }

    // get the required communcation object specialization from the variant based on a list of pattern types
    // - will initialize the communication object if the variant is empty
    // - will throw if a different communication object specialization was initialized earlier
    template<typename PatternList>
    auto& get_co() {
        // extract and deduplicate grids from patterns
        using grids = gridtools::meta::dedup<gridtools::meta::transform<get_grid, PatternList>>;
        // check that all grids are of same type
        static_assert(gridtools::meta::length<grids>::value == 1);

        // extract and deduplicate domain ids from patterns
        using dids = gridtools::meta::dedup<gridtools::meta::transform<get_did, PatternList>>;
        // check that all domain ids are of the same type
        static_assert(gridtools::meta::length<dids>::value == 1);

        // communication object type
        using co_t = ghex::communication_object<gridtools::meta::at_c<grids, 0>, gridtools::meta::at_c<dids, 0>>;

        // check whether co_t is in variant
        static_assert(gridtools::meta::find<communication_object_specializations, co_t>::value <
            gridtools::meta::length<communication_object_specializations>::value);

        // initialize variant with communication object if necessary
        if (m.index() == 0) m.emplace<co_t>(*ctx);

        // return the communication object
        // throws if variant does not hold an alternative of type co_t
        return std::get<co_t>(m);
    }
};

} //namespace regular
} // namespace structured
} // namespace pyghex
