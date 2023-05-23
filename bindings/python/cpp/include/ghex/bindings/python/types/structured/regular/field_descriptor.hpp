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
#ifndef INCLUDED_GHEX_PYBIND_FIELD_DESCRIPTOR_HPP
#define INCLUDED_GHEX_PYBIND_FIELD_DESCRIPTOR_HPP

#include <gridtools/meta.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/bindings/python/type_list.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {
namespace structured {
namespace regular {

namespace detail {
    using field_descriptor_args = gridtools::meta::cartesian_product<
        gridtools::ghex::bindings::python::type_list::data_types,
        gridtools::ghex::bindings::python::type_list::architecture_types,
        // TODO
        gridtools::meta::list<gridtools::ghex::structured::regular::domain_descriptor<int, std::integral_constant<int, 3>>>,
        gridtools::ghex::bindings::python::type_list::layout_maps>;
}

using field_descriptor_specializations = gridtools::meta::transform<gridtools::meta::rename<
    gridtools::ghex::structured::regular::field_descriptor>::template apply,
    detail::field_descriptor_args>;

}
}
}
}
}
}
}
#endif