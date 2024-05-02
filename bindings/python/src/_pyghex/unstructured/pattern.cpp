/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdint>

#include <gridtools/common/for_each.hpp>

#include <ghex/pattern_container.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <context_shim.hpp>
#include <register_class.hpp>
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
            using domain_range = typename type::domain_range;
            using grid_type = ghex::unstructured::grid;
            using pattern_container =
                decltype(ghex::make_pattern<grid_type>(std::declval<ghex::context&>(),
                    std::declval<halo_gen&>(), std::declval<domain_range&>()));
            using fields = field_descriptor_specializations;

            auto _pattern_container = register_class<pattern_container>(m);

            _pattern_container
                .def_property_readonly_static("grid_type", [](const pybind11::object&)
                    { return util::mangle_python<typename pattern_container::grid_type>(); })
                .def_property_readonly_static("domain_id_type", [](const pybind11::object&)
                    { return util::mangle_python<typename pattern_container::domain_id_type>(); });

            m.def(
                "make_pattern_unstructured",
                [](context_shim& c, halo_gen& h, domain_range& d)
                { return ghex::make_pattern<grid_type>(c.m, h, d); },
                pybind11::keep_alive<0, 1>());

            gridtools::for_each<gridtools::meta::transform<gridtools::meta::list, fields>>(
                [&m, &_pattern_container](auto k)
                {
                    using field = gridtools::meta::first<decltype(k)>;
                    // note(stubbiali): pass a lambda function since directly using
                    // `&pattern_container::template operator()<field>` leads to an
                    // "identifier undefined in device code" error when using NVCC
                    _pattern_container.def(
                        "__call__",
                        [](const pattern_container& pattern, field& f)
                        { return pattern(f); },
                        pybind11::keep_alive<0, 2>());
                });

            m.def("expose_cpp_ptr", [](pattern_container* obj){return reinterpret_cast<std::uintptr_t>(obj);});
        });
}

} // namespace unstructured
} // namespace pyghex
