/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <array>
#include <tuple>
#include <string>

#include <gridtools/common/for_each.hpp>

#include <register_class.hpp>
#include <structured/regular/domain_descriptor.hpp>

namespace pyghex
{
namespace structured
{
namespace regular
{
namespace
{
template<std::size_t... I, typename U>
constexpr auto
as_tuple(const U& arr, std::index_sequence<I...>)
{
    return std::make_tuple(arr[I]...);
}

template<typename T, std::size_t N>
constexpr auto
as_tuple(const std::array<T, N>& arr)
{
    return as_tuple(arr, std::make_index_sequence<N>{});
}
} // namespace

void
register_domain_descriptor(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, domain_descriptor_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using domain_id_type = typename type::domain_id_type;
            using dimension = typename type::dimension;
            using array = std::array<int, dimension::value>;

            auto _domain_descriptor = register_class<type>(m);

            _domain_descriptor
                .def(pybind11::init<domain_id_type, array, array>(), "domain_id"_a, "first"_a,
                    "last"_a, "Create a domain descriptor")
                .def("domain_id", &type::domain_id, "Returns the domain id")
                .def(
                    "first", [](const type& d) { return as_tuple(d.first()); },
                    "Returns first coordinate")
                .def(
                    "last", [](const type& d) { return as_tuple(d.last()); },
                    "Returns last coordinate");
        });
}

} // namespace regular
} // namespace structured
} // namespace pyghex
