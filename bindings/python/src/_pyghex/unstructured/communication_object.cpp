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

#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <context_shim.hpp>
#include <register_class.hpp>
#include <unstructured/field_descriptor.hpp>
#include <unstructured/communication_object.hpp>

namespace pyghex
{
namespace unstructured
{
namespace
{
#ifdef GHEX_CUDACC
cudaStream_t
extract_cuda_stream(pybind11::object py_stream)
{
    static_assert(std::is_pointer<cudaStream_t>::value);
    if (py_stream.is_none())
    {
        //NOTE: This is very C++ like, maybe remove and consider as an error?
        return static_cast<cudaStream_t>(nullptr);
    }
    else
    {
        if (pybind11::hasattr(py_stream, "__cuda_stream__"))
        {
            //CUDA stream protocol: https://nvidia.github.io/cuda-python/cuda-core/latest/interoperability.html#cuda-stream-protocol
            pybind11::tuple cuda_stream_protocol =
                pybind11::getattr(py_stream, "__cuda_stream__")();
            if (cuda_stream_protocol.size() != 2)
            {
                std::stringstream error;
                error << "Expected a tuple of length 2, but got one with length "
                      << cuda_stream_protocol.size();
                throw pybind11::type_error(error.str());
            }

            const auto protocol_version = cuda_stream_protocol[0].cast<std::size_t>();
            if (protocol_version == 0)
            {
                std::stringstream error;
                error << "Expected `__cuda_stream__` protocol version 0, but got "
                      << protocol_version;
                throw pybind11::type_error(error.str());
            };

            //Is allowed to be `0`.
            const auto stream_address = cuda_stream_protocol[1].cast<std::uintptr_t>();
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        else if (pybind11::hasattr(py_stream, "ptr"))
        {
            // CuPy stream: See https://docs.cupy.dev/en/latest/reference/generated/cupy.cuda.Stream.html#cupy-cuda-stream
            std::uintptr_t stream_address = py_stream.attr("ptr").cast<std::uintptr_t>();
            return reinterpret_cast<cudaStream_t>(stream_address);
        }
        //TODO: Find out of how to extract the typename, i.e. `type(py_stream).__name__`.
        std::stringstream error;
        error << "Failed to convert the stream object into a CUDA stream.";
        throw pybind11::type_error(error.str());
    };
};
#endif
} // namespace

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
#ifdef GHEX_CUDACC
                .def(
                    "schedule_wait", [](typename type::handle_type& h, pybind11::object py_stream)
                    { return h.schedule_wait(extract_cuda_stream(py_stream)); },
                    pybind11::keep_alive<0, 1>())
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
                            pybind11::keep_alive<0, 1>())
                        .def(
                            "exchange", [](type& co, buffer_info_type& b)
                            { return co.exchange(b); }, pybind11::keep_alive<0, 1>())
                        .def(
                            "exchange", [](type& co, buffer_info_type& b0, buffer_info_type& b1)
                            { return co.exchange(b0, b1); }, pybind11::keep_alive<0, 1>())
                        .def(
                            "exchange",
                            [](type& co, buffer_info_type& b0, buffer_info_type& b1,
                                buffer_info_type& b2) { return co.exchange(b0, b1, b2); },
                            pybind11::keep_alive<0, 1>())
#ifdef GHEX_CUDACC
                        .def(
                            "schedule_exchange",
                            [](type& co,
                                //This should be okay with reference counting?
                                pybind11::object python_stream, std::vector<buffer_info_type> b) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream),
                                    b.begin(), b.end());
                            },
                            pybind11::keep_alive<0, 1>(), pybind11::arg("stream"),
                            pybind11::arg("patterns"))
                        .def(
                            "schedule_exchange",
                            [](type& co, pybind11::object python_stream, buffer_info_type& b)
                            { return co.schedule_exchange(extract_cuda_stream(python_stream), b); },
                            pybind11::keep_alive<0, 1>(), pybind11::arg("stream"),
                            pybind11::arg("b"))
                        .def(
                            "schedule_exchange",
                            [](type& co, pybind11::object python_stream, buffer_info_type& b0,
                                buffer_info_type& b1) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1);
                            },
                            pybind11::keep_alive<0, 1>(), pybind11::arg("stream"),
                            pybind11::arg("b0"), pybind11::arg("b1"))
                        .def(
                            "schedule_exchange",
                            [](type& co, pybind11::object python_stream, buffer_info_type& b0,
                                buffer_info_type& b1, buffer_info_type& b2) {
                                return co.schedule_exchange(extract_cuda_stream(python_stream), b0,
                                    b1, b2);
                            },
                            pybind11::keep_alive<0, 1>(), pybind11::arg("stream"),
                            pybind11::arg("b0"), pybind11::arg("b1"), pybind11::arg("b2"))
#endif
                        ;
                });

            m.def(
                "make_co_unstructured", [](context_shim& c) { return type{c.m}; },
                pybind11::keep_alive<0, 1>());

            m.def("expose_cpp_ptr",
                [](type* obj) { return reinterpret_cast<std::uintptr_t>(obj); });
        });
}

} // namespace unstructured
} // namespace pyghex
