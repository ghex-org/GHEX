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

#include <ghex/pattern_container.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <context_shim.hpp>
#include <util/demangle.hpp>
#include <unstructured/field_descriptor.hpp>
#include <unstructured/pattern.hpp>

namespace pyghex
{
namespace unstructured
{

void
register_pattern(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, make_pattern_traits_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using halo_gen = typename type::halo_gen;
            //using domain_desc = typename type::domain_desc;
            using domain_range = typename type::domain_range;
            using grid_type = ghex::unstructured::grid;
            using pattern_container =
                decltype(ghex::make_pattern<grid_type>(std::declval<ghex::context&>(),
                    std::declval<halo_gen&>(), std::declval<domain_range&>()));
            using fields = field_descriptor_specializations;

            auto pattern_container_name = util::demangle<pattern_container>();
            auto cls = pybind11::class_<pattern_container>(m, pattern_container_name.c_str());
            cls.def_property_readonly_static("__cpp_type__",
                [pattern_container_name](const pybind11::object&)
                { return pattern_container_name; });
            cls.def_property_readonly_static("grid_type", [](const pybind11::object&)
                { return util::demangle<typename pattern_container::grid_type>(); });
            cls.def_property_readonly_static("domain_id_type", [](const pybind11::object&)
                { return util::demangle<typename pattern_container::domain_id_type>(); });

            m.def(
                "make_pattern_unstructured",
                [](context_shim& c, halo_gen& h, domain_range& d)
                { return ghex::make_pattern<grid_type>(c.m, h, d); },
                pybind11::keep_alive<0, 1>());

            gridtools::for_each<gridtools::meta::transform<gridtools::meta::list, fields>>(
                [&m, &cls](auto k)
                {
                    using field = gridtools::meta::first<decltype(k)>;
                    cls.def("__call__", &pattern_container::template operator()<field>,
                        pybind11::keep_alive<0, 2>());
                });
        });
}

} // namespace unstructured
} // namespace pyghex
