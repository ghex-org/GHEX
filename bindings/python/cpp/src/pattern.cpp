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
#include <utility>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/pattern.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/pattern.hpp>

namespace py = pybind11;


template<typename T>
struct type_container {
    using type = T;
};

template <typename pattern_container_type>
struct pattern_container_exporter {
    using pattern_container_trait = gridtools::ghex::bindings::python::types::pattern_container_trait<pattern_container_type>;

    void operator() (pybind11::module_& m, py::class_<pattern_container_type> cls) {
        cls.def_property_readonly_static("grid_type", [] (const pybind11::object&) {
            return gridtools::ghex::bindings::python::utils::demangle<typename pattern_container_type::grid_type>();
        });
        cls.def_property_readonly_static("domain_id_type", [] (const pybind11::object&) {
            return gridtools::ghex::bindings::python::utils::demangle<typename pattern_container_type::domain_id_type>();
        });
        std::cout << "PatternContainer: " << gridtools::ghex::bindings::python::utils::demangle<pattern_container_type>() << std::endl;
        auto make_pattern_wrapper = [] (
                typename pattern_container_trait::context_type& context,
                typename pattern_container_trait::halo_generator_type& hgen,
                typename pattern_container_trait::domain_range_type& d_range) {
            return gridtools::ghex::make_pattern<typename pattern_container_trait::grid_type>(context, hgen, d_range);
        };
        // TODO: specialize name
        m.def("make_pattern", make_pattern_wrapper);

        gridtools::for_each<gridtools::meta::transform<type_container,
                            typename pattern_container_trait::field_descriptor_types>>([&m, &cls] (auto type_container) {
            using field_type = typename decltype(type_container)::type;
            cls.def("__call__", &pattern_container_type::template operator()<field_type>);
        });
    }
};

GHEX_PYBIND11_EXPORT_TYPE(pattern_container_exporter, gridtools::ghex::bindings::python::types::pattern_container_specializations)