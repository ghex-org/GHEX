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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gridtools/common/for_each.hpp>

#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <context_shim.hpp>
#include <util/demangle.hpp>
#include <unstructured/field_descriptor.hpp>
#include <unstructured/communication_object.hpp>

namespace pyghex
{
namespace unstructured
{
void
register_communication_object(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, communication_object_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            auto name = util::demangle<type>();
            auto cls = pybind11::class_<type>(m, name.c_str());

            cls.def_property_readonly_static("__cpp_type__",
                [name](const pybind11::object&) { return name; });

            using handle = typename type::handle_type;
            auto handle_name = util::demangle<handle>();
            pybind11::class_<handle>(m, handle_name.c_str())
                .def("wait", &handle::wait)
                .def("is_ready", &handle::is_ready)
                .def("progress", &handle::progress)
                .def_property_readonly_static("__cpp_type__",
                    [handle_name](const pybind11::object&) { return handle_name; });
            ;

            using pattern_type = typename type::pattern_type;
            using fields = field_descriptor_specializations;

            gridtools::for_each<gridtools::meta::transform<gridtools::meta::list, fields>>(
                [&m, &cls](auto k)
                {
                    using field = gridtools::meta::first<decltype(k)>;
                    using arch_type = typename field::arch_type;
                    using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, field>;

                    cls.def(
                        "exchange",
                        [](type& co, std::vector<buffer_info_type> b)
                        { return co.exchange(b.begin(), b.end()); },
                        pybind11::keep_alive<0, 1>());

                    cls.def(
                        "exchange", [](type& co, buffer_info_type& b) { return co.exchange(b); },
                        pybind11::keep_alive<0, 1>());

                    cls.def(
                        "exchange",
                        [](type& co, buffer_info_type& b0, buffer_info_type& b1)
                        { return co.exchange(b0, b1); },
                        pybind11::keep_alive<0, 1>());

                    cls.def(
                        "exchange",
                        [](type& co, buffer_info_type& b0, buffer_info_type& b1,
                            buffer_info_type& b2) { return co.exchange(b0, b1, b2); },
                        pybind11::keep_alive<0, 1>());
                });

            m.def("make_co_unstructured",
                [](context_shim& c)
                {
                    return type{c.m};
                },
                pybind11::keep_alive<0, 1>());
            
            m.def("expose_cpp_ptr", [](type* obj){return reinterpret_cast<std::uintptr_t>(obj);});
        });
}

} // namespace unstructured
} // namespace pyghex

