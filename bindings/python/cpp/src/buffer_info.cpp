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
#include <ghex/bindings/python/types/buffer_info.hpp>

namespace py = pybind11;

template <typename buffer_info_type>
struct buffer_info_exporter {
    void operator() (pybind11::module_&, py::class_<buffer_info_type>) {}
};

GHEX_PYBIND11_EXPORT_TYPE(buffer_info_exporter, gridtools::ghex::bindings::python::types::buffer_info_specializations)