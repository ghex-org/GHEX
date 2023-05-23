/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#include <array>
#include <tuple>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <util/demangle.hpp>
#include <structured/regular/halo_generator.hpp>

namespace pyghex
{
namespace structured
{
namespace regular
{

void
register_halo_generator(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, halo_generator_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using dimension = typename type::dimension;
            using array = std::array<int, dimension::value>;
            using halo_array = std::array<int, 2 * dimension::value>;
            using periodic_array = std::array<bool, dimension::value>;

            auto type_name = util::demangle<type>();
            pybind11::class_<type>(m, type_name.c_str())
                .def(pybind11::init<array, array, halo_array, periodic_array>(), "first"_a,
                    "last"_a, "halos"_a, "periodic"_a, "Create a halo generator")
                .def("__call__", &type::operator())
                .def_property_readonly_static("__cpp_type__",
                    [type_name](const pybind11::object&) { return type_name; });

            using box2 = typename type::box2;
            auto box2_name = util::demangle<box2>();
            pybind11::class_<box2>(m, box2_name.c_str())
                .def_property_readonly("local",
                    pybind11::overload_cast<>(&box2::local, pybind11::const_))
                .def_property_readonly("global_",
                    pybind11::overload_cast<>(&box2::global, pybind11::const_))
                .def_property_readonly_static("__cpp_type__",
                    [box2_name](const pybind11::object&) { return box2_name; });

            using box = typename type::box;
            auto box_name = util::demangle<box>();
            pybind11::class_<box>(m, box_name.c_str())
                .def_property_readonly("first",
                    [](const box& b)
                    {
                        auto first = b.first();
                        return static_cast<typename decltype(first)::array_type>(first);
                    })
                .def_property_readonly("last",
                    [](const box& b)
                    {
                        auto last = b.last();
                        return static_cast<typename decltype(last)::array_type>(last);
                    })
                .def_property_readonly_static("__cpp_type__",
                    [box_name](const pybind11::object&) { return box_name; });
        });
}

} // namespace regular
} // namespace structured
} // namespace pyghex
