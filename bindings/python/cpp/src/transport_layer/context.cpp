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

#ifdef GHEX_ENABLE_MPI4PY
#include <exception>
#include <mpi4py/mpi4py.h>
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/transport_layer/mpi/context.hpp>
#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/type_list.hpp>
#include <ghex/bindings/python/types/transport_layer/context.hpp>

namespace py = pybind11;

template<typename context_type>
struct context_exporter {
    void operator() (py::module_&, py::class_<context_type> context_cls) {
        context_cls
#ifdef GHEX_ENABLE_MPI4PY
            .def(py::init([] (py::object& py_comm_obj) {
                import_mpi4py();
                if (!PyObject_TypeCheck(py_comm_obj.ptr(), &PyMPIComm_Type)) {
                    std::stringstream ss;
                    ss << "Argument must be `mpi4py.MPI.Comm`, not `" << py_comm_obj.get_type() << "`" << std::endl;
                    throw pybind11::type_error(ss.str());
                }
                MPI_Comm mpi_comm = *PyMPIComm_Get(py_comm_obj.ptr());

                return gridtools::ghex::tl::context_factory<typename context_type::tag>::create(mpi_comm);  }))
#else
            .def(py::init([] (py::object&) {
                throw std::runtime_error("Context construction reqires bindings to be compiled with mpi4py support.");

                return static_cast<context_type*>(nullptr); // just to make pybind11 happy
            }))
#endif
            .def("rank", &context_type::rank)
            .def("size", &context_type::size)
            .def("get_communicator", &context_type::get_communicator);
    }
};

GHEX_PYBIND11_EXPORT_TYPE(context_exporter, gridtools::ghex::bindings::python::types::transport_layer::context_specializations)