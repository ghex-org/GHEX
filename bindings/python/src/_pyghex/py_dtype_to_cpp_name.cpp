/*
 * ghex-org
 *
 * Copyright (c) 2014-2025, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <string>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <gridtools/meta.hpp>
#include <gridtools/common/for_each.hpp>

#include <types.hpp>
#include <util/demangle.hpp>

namespace pyghex
{
std::string
py_dtype_to_cpp_name(nanobind::ndarray<> array)
{
    std::string cpp_name;

    gridtools::for_each<pyghex::types::data>(
        [&cpp_name, &array](auto l)
        {
            using type = decltype(l);

            if (array.dtype() == nanobind::dtype<type>())
            {
                assert(cpp_name.empty());
                cpp_name = util::mangle_python<type>();
            }
        });

    if (cpp_name.empty()) { throw std::invalid_argument("Unsupported numpy dtype"); }

    return cpp_name;
}

void
register_py_dtype_to_cpp_name(nanobind::module_& m)
{
    m.def("py_dtype_to_cpp_name", &py_dtype_to_cpp_name, "Convert numpy dtype to C++ type name");
}

} // namespace pyghex
