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

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

#include <ghex/config.hpp>

namespace pyghex
{

// Returns a python dictionary that python users can use to look up
// which options the GHEX library was configured with at compile time.

nanobind::dict
config()
{
#define mk_str(x) mk_tok(x)
#define mk_tok(x) #x

    nanobind::dict dict;

    dict[nanobind::str("transport")] = nanobind::str(mk_str(GHEX_TRANSPORT_BACKEND));

#ifdef GHEX_USE_GPU
    dict[nanobind::str("gpu")] = nanobind::bool_(true);
#else
    dict[nanobind::str("gpu")] = nanobind::bool_(false);
#endif
    dict[nanobind::str("gpu_mode")] = nanobind::str(mk_str(GHEX_GPU_MODE));

#ifdef GHEX_USE_XPMEM
    dict[nanobind::str("xpmem")] = nanobind::bool_(true);
#else
    dict[nanobind::str("xpmem")] = nanobind::bool_(false);
#endif

#define mk_ver(M, m, p) mk_tok(M) "." mk_tok(m) "." mk_tok(p)
    {
        const char* version = mk_ver(GHEX_VERSION_MAJOR, GHEX_VERSION_MINOR, GHEX_VERSION_PATCH);
        dict[nanobind::str("version")] = nanobind::str(version);
    }
    {
        const char* version = mk_ver(NB_VERSION_MAJOR, NB_VERSION_MINOR, NB_VERSION_PATCH);
        dict[nanobind::str("nanobind-version")] = nanobind::str(version);
    }
#undef mk_str
#undef mk_ver
#undef mk_tok

    return dict;
}

void
print_config(const nanobind::dict& d)
{
    std::stringstream s;
    s << "GHEX's configuration:\n";

    for (auto x : d)
    {
        s << "     " << std::left << std::setw(16) << nanobind::cast<std::string>(x.first) << ": "
          << std::right << std::setw(10) << nanobind::str(x.second).c_str() << "\n";
    }

    nanobind::print(s.str().c_str());
}

// Register configuration
void
register_config(nanobind::module_& m)
{
    m.def("config", &config, "Get GHEX's configuration.")
        .def(
            "print_config", [](const nanobind::dict& d) { return print_config(d); },
            "Print GHEX's configuration.");
}
} // namespace pyghex
