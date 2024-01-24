/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/for_each.hpp>

#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <context_shim.hpp>
#include <register_class.hpp>
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
            using handle = typename type::handle_type;
            using pattern_type = typename type::pattern_type;
            using fields = field_descriptor_specializations;

            auto _communication_object = register_class<type>(m);
            auto _handle = register_class<handle>(m);

            _handle
                .def("wait", &handle::wait)
                .def("is_ready", &handle::is_ready)
                .def("progress", &handle::progress);

            gridtools::for_each<gridtools::meta::transform<gridtools::meta::list, fields>>(
                [&m, &_communication_object](auto k)
                {
                    using field = gridtools::meta::first<decltype(k)>;
                    using arch_type = typename field::arch_type;
                    using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, field>;

                    _communication_object
                        .def(
                            "exchange",
                            [](type& co, std::vector<buffer_info_type> b)
                            { return co.exchange(b.begin(), b.end()); },
                            pybind11::keep_alive<0, 1>())
                        .def(
                            "exchange", [](type& co, buffer_info_type& b) { return co.exchange(b); },
                            pybind11::keep_alive<0, 1>())
                        .def(
                            "exchange",
                            [](type& co, buffer_info_type& b0, buffer_info_type& b1)
                            { return co.exchange(b0, b1); },
                            pybind11::keep_alive<0, 1>())
                        .def(
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
        });
}

} // namespace unstructured
} // namespace pyghex

