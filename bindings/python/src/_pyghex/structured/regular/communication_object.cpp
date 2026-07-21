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
#include <ghex/structured/pattern.hpp>

#include <context_shim.hpp>
#include <cuda_stream.hpp>
#include <register_class.hpp>
#include <structured/regular/field_descriptor.hpp>
#include <structured/regular/communication_object.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/vector.h>
#include <nanobind/stl/string.h>

namespace pyghex
{
namespace structured
{
namespace regular
{

void
register_communication_object(nanobind::module_& m)
{
    auto _communication_object = register_class<communication_object_shim>(m);

    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, communication_object_specializations>>(
        [&m, &_communication_object](auto l)
        {
            using namespace std::string_literals;
            using namespace nanobind::literals;
            using communication_object_type = gridtools::meta::first<decltype(l)>;
            using handle_type = typename communication_object_type::handle_type;
            using pattern_type = typename communication_object_type::pattern_type;
            using dimension = typename communication_object_type::grid_type::dimension;
            using fields = field_descriptor_specializations_<dimension::value>;

            auto _handle = register_class<handle_type>(m);

            _handle
                .def("wait", &handle_type::wait)
#if defined(GHEX_CUDACC)
                .def(
                    "schedule_wait", [](handle_type& h, nanobind::object python_stream)
                    { return h.schedule_wait(extract_cuda_stream(python_stream)); },
                    nanobind::arg("stream").none())
#endif
                .def("is_ready", &handle_type::is_ready)
                .def("progress", &handle_type::progress);

            gridtools::for_each<gridtools::meta::transform<gridtools::meta::list, fields>>(
                [&m, &_communication_object](auto k)
                {
                    using field = gridtools::meta::first<decltype(k)>;
                    using arch_type = typename field::arch_type;
                    using buffer_info_type = ghex::buffer_info<pattern_type, arch_type, field>;

                    _communication_object
                        .def(
                            "exchange",
                            [](communication_object_shim& co, std::vector<buffer_info_type> b)
                            { return co.exchange(b.begin(), b.end()); },
                            nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange", [](communication_object_shim& co, buffer_info_type& b)
                            { return co.exchange(b); }, nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange",
                            [](communication_object_shim& co, buffer_info_type& b0,
                                buffer_info_type& b1) { return co.exchange(b0, b1); },
                            nanobind::keep_alive<0, 1>())
                        .def(
                            "exchange",
                            [](communication_object_shim& co, buffer_info_type& b0,
                                buffer_info_type& b1, buffer_info_type& b2)
                            { return co.exchange(b0, b1, b2); },
                            nanobind::keep_alive<0, 1>())
#if defined(GHEX_CUDACC)
                        .def(
                            "schedule_exchange",
                            [](communication_object_shim& co, nanobind::object python_stream,
                                std::vector<buffer_info_type> b) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream),
                                    b.begin(), b.end());
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("patterns"))
                        .def(
                            "schedule_exchange",
                            [](communication_object_shim& co, nanobind::object python_stream,
                                buffer_info_type& b)
                            { return co.schedule_exchange(extract_cuda_stream(python_stream), b); },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b"))
                        .def(
                            "schedule_exchange",
                            [](communication_object_shim& co, nanobind::object python_stream,
                                buffer_info_type& b0, buffer_info_type& b1) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1);
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b0"), nanobind::arg("b1"))
                        .def(
                            "schedule_exchange",
                            [](communication_object_shim& co, nanobind::object python_stream,
                                buffer_info_type& b0, buffer_info_type& b1, buffer_info_type& b2) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1, b2);
                            },
                            nanobind::keep_alive<0, 1>(), nanobind::arg("stream").none(),
                            nanobind::arg("b0"), nanobind::arg("b1"), nanobind::arg("b2"))
                        .def("has_scheduled_exchange", [](communication_object_shim& co) -> bool
                            { return co.has_scheduled_exchange(); })
#endif // end scheduled exchange
                        ;
                });
        });

    m.def(
        "make_co_regular",
        [](context_shim& c) { return communication_object_shim{&c.m, std::monostate{}}; },
        nanobind::keep_alive<0, 1>());
}

} //namespace regular
} // namespace structured
} // namespace pyghex
