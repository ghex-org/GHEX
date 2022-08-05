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

#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <gridtools/meta.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/communication_object.hpp>

namespace py = pybind11;

template<typename T>
struct type_container {
    using type = T;
};

template <typename communication_object_type>
struct communication_object_exporter {
    using communication_object_trait = gridtools::ghex::bindings::python::types::communication_object_trait<
        communication_object_type>;

    void operator() (pybind11::module_& m, py::class_<communication_object_type> cls) {
        cls.def(py::init<typename communication_object_trait::communicator_type>());

        gridtools::for_each<gridtools::meta::transform<type_container,
                            typename communication_object_trait::buffer_info_types>>([&m, &cls] (auto type_container) {
            using buffer_info_type = typename decltype(type_container)::type;
            cls.def("exchange", [] (communication_object_type& co,
                                    buffer_info_type& b) {
                return co.exchange(b);
            });
            cls.def("exchange", [] (communication_object_type& co,
                                    buffer_info_type& b1,
                                    buffer_info_type& b2) {
                return co.exchange(b1, b2);
            });
            cls.def("exchange", [] (communication_object_type& co,
                                    buffer_info_type& b1,
                                    buffer_info_type& b2,
                                    buffer_info_type& b3) {
                return co.exchange(b1, b2, b3);
            });
        });
    }
};

GHEX_PYBIND11_EXPORT_TYPE(communication_object_exporter, gridtools::ghex::bindings::python::types::communication_object_specializations)

template <typename communication_handle_type>
struct communication_handle_exporter {
    void operator() (pybind11::module_&, py::class_<communication_handle_type> cls) {
        cls.def("wait", &communication_handle_type::wait);
    }
};

GHEX_PYBIND11_EXPORT_TYPE(communication_handle_exporter, gridtools::ghex::bindings::python::types::communication_handle_specializations)


