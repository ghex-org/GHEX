/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <iomanip>
#include <ios>
#include <sstream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/config.hpp>

namespace pyghex
{

// Returns a python dictionary that python users can use to look up
// which options the GHEX library was configured with at compile time.

pybind11::dict
config()
{
#define mk_str(x) mk_tok(x)
#define mk_tok(x) #x

    pybind11::dict dict;

#ifdef GHEX_ENABLE_MPI4PY
    dict[pybind11::str("mpi4py")] = pybind11::bool_(true);
#else
    dict[pybind11::str("mpi4py")] = pybind11::bool_(false);
#endif

    dict[pybind11::str("transport")] = pybind11::str(mk_str(GHEX_TRANSPORT_BACKEND));

#ifdef GHEX_USE_GPU
    dict[pybind11::str("gpu")] = pybind11::bool_(true);
#else
    dict[pybind11::str("gpu")] = pybind11::bool_(false);
#endif
    dict[pybind11::str("gpu_mode")] = pybind11::str(mk_str(GHEX_GPU_MODE));

#ifdef GHEX_USE_XPMEM
    dict[pybind11::str("xpmem")] = pybind11::bool_(true);
#else
    dict[pybind11::str("xpmem")] = pybind11::bool_(false);
#endif

#define mk_ver(M, m, p) mk_tok(M) "." mk_tok(m) "." mk_tok(p)
    {
        const char* version = mk_ver(GHEX_VERSION_MAJOR, GHEX_VERSION_MINOR, GHEX_VERSION_PATCH);
        dict[pybind11::str("version")] = pybind11::str(version);
    }
    {
        const char* version =
            mk_ver(PYBIND11_VERSION_MAJOR, PYBIND11_VERSION_MINOR, PYBIND11_VERSION_PATCH);
        dict[pybind11::str("pybind-version")] = pybind11::str(version);
    }
#undef mk_str
#undef mk_ver
#undef mk_tok

    return dict;
}

void
print_config(const pybind11::dict& d)
{
    std::stringstream s;
    s << "GHEX's configuration:\n";

    for (auto x : d)
    {
        s << "     " << std::left << std::setw(16) << x.first << ": " << std::right << std::setw(10)
          << x.second << "\n";
    }

    pybind11::print(s.str());
}

// Register configuration
void
register_config(pybind11::module& m)
{
    m
        .def("config", &config, "Get GHEX's configuration.")
        .def("print_config", [](const pybind11::dict& d) { return print_config(d); },
            "Print GHEX's configuration.");
}
} // namespace pyghex
