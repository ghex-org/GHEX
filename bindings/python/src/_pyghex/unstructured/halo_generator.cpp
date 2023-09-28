/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gridtools/common/for_each.hpp>

#include <util/demangle.hpp>
#include <unstructured/halo_generator.hpp>

namespace pyghex
{
namespace unstructured
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
            using global_index_type = typename type::global_index_type;
            using halo = typename type::halo;

            auto type_name = util::demangle<type>();
            pybind11::class_<type>(m, type_name.c_str())
                .def(pybind11::init<>(), "Create a halo generator")
                .def(pybind11::init([](const std::vector<global_index_type>& gids){ return type{gids};}))
                .def("__call__", &type::operator())
                .def_property_readonly_static("__cpp_type__",
                    [type_name](const pybind11::object&) { return type_name; });

            auto halo_name = util::demangle<halo>();
            pybind11::class_<halo>(m, halo_name.c_str())
                .def_property_readonly_static("__cpp_type__",
                    [halo_name](const pybind11::object&) { return halo_name; });
        });
}

} // namespace unstructured
} // namespace pyghex
