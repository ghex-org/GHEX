/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <cstdint>
#include <pybind11/pybind11.h>

#include <context_shim.hpp>
#include <util/to_string.hpp>

namespace pyghex
{

context_shim::context_shim(mpi_comm_shim mpi_comm_, bool thread_safe)
: m_thread_safe{thread_safe}
, m{mpi_comm_.comm, thread_safe}
, m_mpi_comm{m.mpi_comm()}
{
}

namespace util
{
template<>
std::string
to_string(const context_shim& c)
{
    return std::string("<ghex.context: ") + to_string(c.m_mpi_comm) +
           (c.m_thread_safe ? ", thread safe" : ", not thread safe") + std::string(">");
}
} // namespace util

void
register_context(pybind11::module& m)
{
    using namespace std::string_literals;
    using namespace pybind11::literals;

    pybind11::class_<context_shim>(m, "context")
        .def(pybind11::init<mpi_comm_shim, bool>(), "mpi_comm"_a, "thread_safe"_a = false,
            "Create a ghex context")
        .def("__str__", util::to_string<context_shim>)
        .def("__repr__", util::to_string<context_shim>)
        .def(
            "mpi_comm", [](const context_shim& c) { return c.m_mpi_comm; }, "MPI communicator")
        .def(
            "rank", [](const context_shim& c) { return c.m.rank(); }, "rank of this process")
        .def(
            "size", [](const context_shim& c) { return c.m.size(); },
            "number of ranks within the communicator");

    m.def("expose_cpp_ptr", [](context_shim* obj){return reinterpret_cast<std::uintptr_t>(&obj->m);});
}

} // namespace pyghex
