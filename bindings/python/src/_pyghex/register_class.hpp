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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace pyghex
{

template<typename T>
auto register_class(pybind11::module& m) {

    auto demangled = util::demangle<T>();
    auto pymangled = util::mangle_python(demangled);
    return pybind11::class_<T>(m, pymangled.c_str())
        .def_property_readonly_static("__cpp_type__", [demangled](const pybind11::object&) { return demangled; })
        .def("__str__", [pymangled](const T&) { return "<ghex." + pymangled + ">"; })
        .def("__repr__", [pymangled](const T&) { return "<ghex." + pymangled + ">"; });
}

} // namespace pyghex
