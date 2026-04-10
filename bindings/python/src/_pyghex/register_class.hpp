/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <util/demangle.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>

namespace pyghex
{

template<typename T>
auto
register_class(nanobind::module_& m)
{
    auto demangled = util::demangle<T>();
    auto pymangled = util::mangle_python(demangled);
    return nanobind::class_<T>(m, pymangled.c_str())
        .def_prop_ro_static("__cpp_type__",
            [demangled](const nanobind::object&) { return demangled; })
        .def("__str__", [pymangled](const T&) { return "<ghex." + pymangled + ">"; })
        .def("__repr__", [pymangled](const T&) { return "<ghex." + pymangled + ">"; });
}

} // namespace pyghex
