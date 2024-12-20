/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <stdexcept>
#include <string>
#include <sstream>

#include <mpi_comm_shim.hpp>
#include <util/to_string.hpp>

namespace pyghex
{

struct mpi_error : std::runtime_error
{
    mpi_error(const std::string& what)
    : std::runtime_error{what}
    {
    }
};

namespace
{

// Convert a Python object to an MPI Communicator.
inline MPI_Comm
convert_to_mpi_comm(pybind11::object o)
{
    if (!pybind11::hasattr(o, "py2f"))
        throw pybind11::type_error("Argument must be `mpi4py.MPI.Comm`");
    return MPI_Comm_f2c(o.attr("py2f")().cast<MPI_Fint>());
}

} // anonymous namespace

namespace util
{
template<>
std::string
to_string(const mpi_comm_shim& c)
{
    if (c.comm == MPI_COMM_WORLD) { return "<ghex.mpi_comm: MPI_COMM_WORLD>"; }
    else
    {
        std::ostringstream o;
        o << "<ghex.mpi_comm: " << c.comm << ">";
        return o.str();
    }
}
} // namespace util

mpi_comm_shim::mpi_comm_shim(pybind11::object o) { comm = convert_to_mpi_comm(o); }

void
mpi_init(bool threadsafe)
{
    int provided = MPI_THREAD_SINGLE;
    int requested = threadsafe ? MPI_THREAD_MULTIPLE : MPI_THREAD_SINGLE;
    int ev = MPI_Init_thread(nullptr, nullptr, requested, &provided);
    if (ev) throw mpi_error("MPI_Init_thread");
    else if (provided < requested)
    {
        throw mpi_error("MPI_Init_thread: MPI_THREAD_MULTIPLE unsupported");
    }
}

void
mpi_finalize()
{
    MPI_Finalize();
}

bool
mpi_is_initialized()
{
    int initialized;
    MPI_Initialized(&initialized);
    return initialized;
}

bool
mpi_is_finalized()
{
    int finalized;
    MPI_Finalized(&finalized);
    return finalized;
}

void
register_mpi(pybind11::module& m)
{
    using namespace std::string_literals;
    using namespace pybind11::literals;

    pybind11::class_<mpi_comm_shim> mpi_comm(m, "mpi_comm");
    mpi_comm.def(pybind11::init<>())
        .def(pybind11::init([](pybind11::object o) { return mpi_comm_shim(o); }), "mpi_comm_obj"_a,
            "MPI communicator object.")
        .def("__str__", util::to_string<mpi_comm_shim>)
        .def("__repr__", util::to_string<mpi_comm_shim>);

    m.def("mpi_init", &mpi_init, "thread_safe"_a = true, "Initialize MPI.");
    m.def("mpi_finalize", &mpi_finalize, "Finalize MPI (calls MPI_Finalize)");
    m.def("mpi_is_initialized", &mpi_is_initialized, "Check if MPI is initialized.");
    m.def("mpi_is_finalized", &mpi_is_finalized, "Check if MPI is finalized.");

}

} // namespace pyghex
