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

#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/structured/regular/field_descriptor.hpp>

namespace py = pybind11;

template <int... val>
using int_tuple_constant = gridtools::meta::list<std::integral_constant<int, val>...>;


// todo: dim independent
template<typename T, typename Arch, typename DomainDescriptor, typename Layout>
struct type_exporter<gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, Layout>> {
    using field_descriptor_type = gridtools::ghex::structured::regular::field_descriptor<T, Arch, DomainDescriptor, Layout>;

    void operator() (pybind11::module_&, py::class_<field_descriptor_type> cls) {
        constexpr std::size_t dim = 3;

        using array_type = std::array<int, dim>;

        auto wrapper = [] (const DomainDescriptor& dom, py::buffer b, const array_type& offsets, const array_type& extents) {
            py::buffer_info info = b.request();

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