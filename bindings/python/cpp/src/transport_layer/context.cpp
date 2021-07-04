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

#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <gridtools/meta.hpp>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/bindings/python/utils/type_exporter.hpp>
#include <ghex/bindings/python/type_list.hpp>

#include <ghex/transport_layer/mpi/context.hpp>

namespace py = pybind11;

namespace detail {
    using args = gridtools::meta::cartesian_product<
                        gridtools::meta::list<gridtools::ghex::tl::mpi_tag>>;

    template<typename Transport>
    using context_type = typename gridtools::ghex::tl::context_factory<Transport>::context_type;

    using specializations = gridtools::meta::transform<gridtools::meta::rename<context_type>::template apply, args>;
}

template<typename ContextType>
struct context_type_exporter {
    using context_type = ContextType;

    void operator() (pybind11::module_& m, py::class_<context_type> context_cls) {
        context_cls
#ifdef GHEX_ENABLE_MPI4PY
            .def(py::init([] (pybind11::object& py_comm_obj) {
                import_mpi4py();
                if (!PyObject_TypeCheck(py_comm_obj.ptr(), &PyMPIComm_Type)) {
                    std::stringstream ss;
                    ss << "Argument must be `mpi4py.MPI.Comm`, not `" << py_comm_obj.get_type() << "`" << std::endl;
                    throw pybind11::type_error(ss.str());
                }
                MPI_Comm mpi_comm = *PyMPIComm_Get(py_comm_obj.ptr());

                return gridtools::ghex::tl::context_factory<typename context_type::tag>::create(mpi_comm);  }))
#endif
            .def("rank", &context_type::rank)
            .def("size", &context_type::size)
            .def("get_communicator", &context_type::get_communicator);

        using communicator_type = typename context_type::communicator_type;
        py::class_<communicator_type>(m, "Communicator");
    }
};

GHEX_PYBIND11_EXPORT_TYPE(context_type_exporter, detail::specializations)