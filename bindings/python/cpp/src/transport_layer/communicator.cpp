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

#include <gridtools/meta.hpp>
#include <ghex/transport_layer/mpi/communicator.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/type_list.hpp>

namespace py = pybind11;

template <typename communicator_type>
struct communicator_exporter {
    void operator() (pybind11::module_&, py::class_<communicator_type>) {}
};

GHEX_PYBIND11_EXPORT_TYPE(communicator_exporter, gridtools::meta::list<gridtools::ghex::bindings::python::type_list::communicator_type>)