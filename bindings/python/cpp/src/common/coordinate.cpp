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

#include <ghex/buffer_info.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/types/common/coordinate.hpp>

namespace py = pybind11;

template <typename coordinate_type>
struct coordinate_exporter {
    void operator() (pybind11::module_&, py::class_<coordinate_type>) {}
};

GHEX_PYBIND11_EXPORT_TYPE(coordinate_exporter, gridtools::ghex::bindings::python::types::common::coordinate_specializations)