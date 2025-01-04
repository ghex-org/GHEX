/*
 * ghex-org
 *
 * Copyright (c) 2014-2024, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <string>

#include <pybind11/numpy.h>

#include <gridtools/meta.hpp>
#include <gridtools/common/for_each.hpp>

#include <types.hpp>
#include <util/demangle.hpp>

namespace py = pybind11;

namespace pyghex
{

std::string py_dtype_to_cpp_name(py::dtype dtype) {
    std::string cpp_name;

    gridtools::for_each<pyghex::types::data>([&cpp_name, &dtype](auto l) {
        using type = decltype(l);

        assert(cpp_name.empty());
        if (dtype.is(py::dtype::of<type>())) {
            cpp_name = util::demangle<type>();
        }
    });

    if (cpp_name.empty()) {
        throw std::invalid_argument("Unsupported numpy dtype");
    }

    return cpp_name;
}

void
register_py_dtype_to_cpp_name(pybind11::module& m)
{
    m.def("py_dtype_to_cpp_name", &py_dtype_to_cpp_name, "Convert numpy dtype to C++ type name");
}

} // namespace pyghex