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
#ifndef INCLUDED_GHEX_PYBIND_DOMAIN_DESCRIPTOR_HPP
#define INCLUDED_GHEX_PYBIND_DOMAIN_DESCRIPTOR_HPP

#include <gridtools/meta.hpp>
#include <ghex/common/coordinate.hpp>
#include <ghex/bindings/python/type_list.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {
namespace common {

namespace detail {
    using coordinate_args = gridtools::meta::cartesian_product<
                            gridtools::ghex::bindings::python::type_list::domain_id_types,
                            gridtools::ghex::bindings::python::type_list::dims>;

    template <typename T, typename Dim>
    using coordinate_type = gridtools::ghex::coordinate<std::array<T, Dim::value>>;

}

using coordinate_specializations = gridtools::meta::transform<gridtools::meta::rename<
        detail::coordinate_type>::template apply, detail::coordinate_args>;

}
}
}
}
}
}
#endif