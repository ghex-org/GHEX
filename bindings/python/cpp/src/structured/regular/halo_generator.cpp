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

#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <gridtools/meta.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/type_list.hpp>

#include <ghex/structured/regular/domain_descriptor.hpp>

namespace py = pybind11;

namespace detail {
    using args = gridtools::meta::cartesian_product<
                        gridtools::ghex::bindings::python::type_list::domain_id_types,
                        gridtools::meta::list<std::integral_constant<int, 3>>>;

    template<typename DomainIdType, typename Dimension>
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<DomainIdType, Dimension::value>;

    using specializations = gridtools::meta::transform<gridtools::meta::rename<halo_generator_type>::template apply, args>;
}

template<typename DomainIdType, int Dimension>
struct type_exporter<gridtools::ghex::structured::regular::halo_generator<DomainIdType, Dimension>> {
    using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<DomainIdType, Dimension>;

    void operator() (pybind11::module_& m, py::class_<halo_generator_type> halo_gen_cls) {
        static_assert(std::is_same<DomainIdType, int>::value, "Not implemented. Only integer domain types allowed for now.");

        using dim_array_t = std::array<int, Dimension>;
        using halo_array_t = std::array<int, 2*Dimension>;
        using periodic_array_t  = std::array<bool, Dimension>;

        // halo_generator
        halo_gen_cls.def(py::init<dim_array_t, dim_array_t, halo_array_t, periodic_array_t>())
                        .def("__call__", &halo_generator_type::operator()); // todo

        // box2
        py::class_<typename halo_generator_type::box2>(m, "Box2")
                .def_property_readonly("local", py::overload_cast<>(&halo_generator_type::box2::local, py::const_))
                .def_property_readonly("global_", py::overload_cast<>(&halo_generator_type::box2::global, py::const_));
        py::class_<typename halo_generator_type::box>(m, "Box")
                .def_property_readonly("first", [] (const typename halo_generator_type::box& b) {
                    auto first = b.first();
                    return static_cast<typename decltype(first)::array_type>(first);
                })
                .def_property_readonly("last", [] (const typename halo_generator_type::box& b) {
                    auto last = b.last();
                    return static_cast<typename decltype(last)::array_type>(last);
                });
    }
};

GHEX_PYBIND11_EXPORT_TYPE(type_exporter, detail::specializations)