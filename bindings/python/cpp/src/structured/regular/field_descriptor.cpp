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

#include <cassert>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gridtools/common/stride_util.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/structured/regular/field_descriptor.hpp>

namespace py = pybind11;

using namespace pybind11::literals;

template <int... val>
using int_tuple_constant = gridtools::meta::list<std::integral_constant<int, val>...>;

template <typename Arch>
struct buffer_info_accessor {};

template <>
struct buffer_info_accessor<gridtools::ghex::cpu> {
    static py::buffer_info get(py::object& buffer) {
        return buffer.cast<py::buffer>().request();
    }
};

template <>
struct buffer_info_accessor<gridtools::ghex::gpu> {
    static py::buffer_info get(py::object& buffer) {
        py::dict info = buffer.attr("__cuda_array_interface__");

        bool readonly = info["data"].cast<py::tuple>()[1].cast<bool>();
        assert(!readonly);

        void* ptr = reinterpret_cast<void*>(info["data"].cast<py::tuple>()[0].cast<py::ssize_t>());

        // create buffer protocol format and itemsize from typestr
        py::function memory_view = py::module::import("builtins").attr("memoryview");
        py::function np_array = py::module::import("numpy").attr("array");
        py::buffer empty_buffer = memory_view(np_array(py::list(), "dtype"_a=info["typestr"]));
        py::ssize_t itemsize = empty_buffer.request().itemsize;
        std::string format = empty_buffer.request().format;

        std::vector<py::ssize_t> shape = info["shape"].cast<std::vector<py::ssize_t>>();
        py::ssize_t ndim = shape.size();

        std::vector<py::ssize_t> strides(ndim);
        if (py::isinstance<py::none>(info["strides"])) {
            strides[ndim-1] = 1;
            for (int i=ndim-2; i>=0; --i) {
                strides[i] = (strides[i+1]*shape[i+1]) * itemsize;
            }
        } else {
            strides = info["strides"].cast<std::vector<py::ssize_t>>();
            assert(strides.size() == ndim);
        }

        return py::buffer_info(
            ptr,        /* Pointer to buffer */
            itemsize,   /* Size of one scalar */
            format,     /* Python struct-style format descriptor */
            ndim,       /* Number of dimensions */
            shape,      /* Buffer dimensions */
            strides     /* Strides (in bytes) for each index */
        );
    }
};

template <typename Arch>
py::buffer_info get_buffer_info(py::object& buffer) {
    return buffer_info_accessor<Arch>::get(buffer);
}


// todo: dim independent
template<typename T, typename Arch, typename DomainDescriptor, typename Layout>
struct type_exporter<gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, Layout>> {
    using field_descriptor_type = gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, Layout>;

    void operator() (pybind11::module_&, py::class_<field_descriptor_type> cls) {
        constexpr std::size_t dim = 3;

        using array_type = std::array<int, dim>;

        auto wrapper = [] (const DomainDescriptor& dom, py::object& b, const array_type& offsets, const array_type& extents) {
            py::buffer_info info = get_buffer_info<Arch>(b);

            if (info.format != py::format_descriptor<T>::format()) {
                std::stringstream error;
                error << "Incompatible format: expected a " << typeid(T).name() << " buffer.";
                throw pybind11::type_error(error.str());
            }
            std::array<int, 3> buffer_order = {0, 1, 2};
            std::sort(buffer_order.begin(), buffer_order.end(), [&info](int a, int b) {
                return info.strides[a] > info.strides[b];
            });
            for (size_t i=0; i<3; ++i) {
                if (buffer_order[i] != Layout::at(i)) {
                    throw pybind11::type_error("Buffer has a different layout than specified.");
                }
            }

            if (info.ndim != 3)
                throw std::runtime_error("Incompatible buffer dimension.");

            return gridtools::ghex::wrap_field<Arch, Layout>(dom, static_cast<T*>(info.ptr), offsets, extents);
        };

        cls.def(py::init(wrapper));
    }
};

GHEX_PYBIND11_EXPORT_TYPE(type_exporter, gridtools::ghex::bindings::python::types::structured::regular::field_descriptor_specializations)