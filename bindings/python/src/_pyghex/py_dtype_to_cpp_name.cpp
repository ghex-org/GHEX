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

#include <Python.h>

#include <gridtools/meta.hpp>
#include <gridtools/common/for_each.hpp>

#include <types.hpp>
#include <util/demangle.hpp>

namespace nb = nanobind;

namespace pyghex
{

namespace
{
nb::object
numpy_dtype(const nb::handle& value)
{
    static nb::module_ numpy = nb::module_::import_("numpy");
    return numpy.attr("dtype")(nb::borrow<nb::object>(value));
}

template<typename T>
nb::object
dtype_of()
{
    if constexpr (std::is_same_v<T, double>) return numpy_dtype(nb::str("float64"));
    else if constexpr (std::is_same_v<T, float>)
        return numpy_dtype(nb::str("float32"));
    else if constexpr (std::is_same_v<T, bool>)
        return numpy_dtype(nb::str("bool"));
    else if constexpr (std::is_same_v<T, std::int32_t>)
        return numpy_dtype(nb::str("int32"));
    else if constexpr (std::is_same_v<T, std::int64_t>)
        return numpy_dtype(nb::str("int64"));
    else
        static_assert(sizeof(T) == 0, "unsupported dtype");
}
} // namespace

std::string
py_dtype_to_cpp_name(nb::handle dtype)
{
    const auto  canonical_dtype = numpy_dtype(dtype);
    std::string cpp_name;

    gridtools::for_each<pyghex::types::data>(
        [&cpp_name, &canonical_dtype](auto l)
        {
            using type = decltype(l);

            auto candidate_dtype = dtype_of<type>();
            if (PyObject_RichCompareBool(canonical_dtype.ptr(), candidate_dtype.ptr(), Py_EQ) == 1)
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
