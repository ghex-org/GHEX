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
#include <sstream>

#include <gridtools/common/for_each.hpp>

#include <ghex/buffer_info.hpp>
#include <ghex/unstructured/pattern.hpp>

#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <context_shim.hpp>
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
namespace
{
#if defined(GHEX_CUDACC)
cudaStream_t
extract_cuda_stream(nanobind::object python_stream)
{
    static_assert(std::is_pointer<cudaStream_t>::value);
    if (python_stream.is_none())
    {
        // NOTE: This is very C++ like, maybe remove and consider as an error?
        return static_cast<cudaStream_t>(nullptr);
    }
    else
    {
        if (nanobind::hasattr(python_stream, "__cuda_stream__"))
        {
            // CUDA stream protocol: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
            nanobind::tuple cuda_stream_protocol =
                nanobind::cast<nanobind::tuple>(python_stream.attr("__cuda_stream__")());
            if (cuda_stream_protocol.size() != 2)
            {
                std::stringstream error;
                error << "Expected a tuple of length 2, but got one with length "
                      << cuda_stream_protocol.size();
                throw nanobind::type_error(error.str().c_str());
            }

            const auto protocol_version = nanobind::cast<std::size_t>(cuda_stream_protocol[0]);
            if (protocol_version != 0)
            {
                std::stringstream error;
                error << "Expected `__cuda_stream__` protocol version 0, but got "
                      << protocol_version;
                throw nanobind::type_error(error.str().c_str());
            }

            const auto stream_address = nanobind::cast<std::uintptr_t>(cuda_stream_protocol[1]);
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        else if (nanobind::hasattr(python_stream, "ptr"))
        {
            // CuPy stream: See https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Stream.html#cupy-cuda-stream
            std::uintptr_t stream_address =
                nanobind::cast<std::uintptr_t>(python_stream.attr("ptr"));
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        // TODO: Find out of how to extract the typename, i.e. `type(python_stream).__name__`.
        std::stringstream error;
        error << "Failed to convert the stream object into a CUDA stream.";
        throw nanobind::type_error(error.str().c_str());
    }
}
#endif
} // namespace

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
