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

#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/structured/regular/domain_descriptor.hpp>

namespace py = pybind11;

template<std::size_t... I, typename U>
constexpr auto as_tuple(const U &arr, std::index_sequence<I...>) {
    return std::make_tuple(arr[I]...);
}

template<typename T, std::size_t N>
constexpr auto as_tuple(const std::array<T, N> &arr) {
    return as_tuple(arr, std::make_index_sequence<N>{});
}

template<typename DomainIdType, typename Dimension_t>
struct type_exporter<gridtools::ghex::structured::regular::domain_descriptor<DomainIdType, Dimension_t>> {
    static constexpr int dimension = Dimension_t::value;

    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<DomainIdType, Dimension_t>;

    void operator() (pybind11::module_&, py::class_<domain_descriptor_type> cls) {
        using dim_array_t = std::array<int, dimension>;

        cls.def(py::init<DomainIdType, dim_array_t, dim_array_t>())
            .def("domain_id", &domain_descriptor_type::domain_id)
            .def("first", [] (const domain_descriptor_type& domain_desc) { return as_tuple(domain_desc.first()); })
            .def("last", [] (const domain_descriptor_type& domain_desc) { return as_tuple(domain_desc.last()); });
    }
};

GHEX_PYBIND11_EXPORT_TYPE(type_exporter, gridtools::ghex::bindings::python::types::structured::regular::domain_descriptor_specializations)