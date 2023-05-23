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

//#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <gridtools/common/for_each.hpp>
#include <gridtools/meta.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace utils {

template <template <typename...> class WrapperRegistrator, typename... Args>
void register_wrapper(pybind11::module& m) {
    using namespace gridtools;
    using register_instantiations = meta::transform<
        meta::rename<WrapperRegistrator>::template apply,
        meta::cartesian_product<Args...>
    >;

    for_each<register_instantiations>([&m] (auto registrator) {
        registrator(m);
    });
}

} // namespace gridtools
} // namespace ghex
} // namespace bindings
} // namespace python
} // namespace utils
