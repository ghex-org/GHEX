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

#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/pattern.hpp>

#include <context_shim.hpp>
#include <cuda_stream.hpp>
#include <register_class.hpp>
#include <unstructured/field_descriptor.hpp>
#include <unstructured/communication_object.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace pyghex
{
namespace unstructured
{

void
register_communication_object(nanobind::module_& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, communication_object_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace nanobind::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using handle = typename type::handle_type;
            using pattern_type = typename type::pattern_type;
            using fields = field_descriptor_specializations;

            auto _communication_object = register_class<type>(m);
            auto _handle = register_class<handle>(m);

            _handle
                .def("wait", &handle::wait)
#if defined(GHEX_CUDACC)
                .def(
                    "schedule_wait",
                    [](typename type::handle_type& h, nanobind::object python_stream)
                    { return h.schedule_wait(extract_cuda_stream(python_stream)); },
                    nanobind::arg("stream").none())
#endif
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
                            "exchange", [](type& co, std::vector<buffer_info_type> b)
                            { return co.exchange(b.begin(), b.end()); },
                            nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange", [](type& co, buffer_info_type& b)
                            { return co.exchange(b); }, nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange", [](type& co, buffer_info_type& b0, buffer_info_type& b1)
                            { return co.exchange(b0, b1); }, nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange",
                            [](type& co, buffer_info_type& b0, buffer_info_type& b1,
                                buffer_info_type& b2) { return co.exchange(b0, b1, b2); },
                            nanobind::keep_alive<0, 1>())
#if defined(GHEX_CUDACC)
                        .def(
                            "schedule_exchange",
                            [](type& co, nanobind::object python_stream,
                                std::vector<buffer_info_type> b) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream),
                                    b.begin(), b.end());
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("patterns"))
                        .def(
                            "schedule_exchange",
                            [](type& co, nanobind::object python_stream, buffer_info_type& b)
                            { return co.schedule_exchange(extract_cuda_stream(python_stream), b); },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b"))
                        .def(
                            "schedule_exchange",
                            [](type& co, nanobind::object python_stream, buffer_info_type& b0,
                                buffer_info_type& b1) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1);
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b0"), nanobind::arg("b1"))
                        .def(
                            "schedule_exchange",
                            [](type& co, nanobind::object python_stream, buffer_info_type& b0,
                                buffer_info_type& b1, buffer_info_type& b2) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1, b2);
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b0"), nanobind::arg("b1"), nanobind::arg("b2"))
                        .def("has_scheduled_exchange",
                            [](type& co) -> bool { return co.has_scheduled_exchange(); })
#endif // end scheduled exchange
                        ;
                });

            m.def(
                "make_co_unstructured", [](context_shim& c) { return type{c.m}; },
                nanobind::keep_alive<0, 1>());

            m.def("expose_cpp_ptr",
                [](type* obj) { return reinterpret_cast<std::uintptr_t>(obj); });
        });
}

} // namespace unstructured
} // namespace pyghex
