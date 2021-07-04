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
#pragma once

#include <cxxabi.h>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace utils {

template <typename T>
std::string demangle() {
    int status = 0;
    std::unique_ptr<char, void (*)(void *)> res {
        abi::__cxa_demangle(typeid(T).name(), nullptr, nullptr, &status), std::free };
    if (status != 0)
        throw std::runtime_error("Could not demangle.");

    return res.get();
}

} // namespace gridtools
} // namespace ghex
} // namespace bindings
} // namespace python
} // namespace utils