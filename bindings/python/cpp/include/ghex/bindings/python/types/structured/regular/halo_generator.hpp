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
#ifndef INCLUDED_GHEX_PYBIND_HALO_GENERATOR_HPP
#define INCLUDED_GHEX_PYBIND_HALO_GENERATOR_HPP

#include <gridtools/meta.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/bindings/python/type_list.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {
namespace structured {
namespace regular {

namespace detail {
    using halo_generator_args = gridtools::meta::cartesian_product<
                        gridtools::ghex::bindings::python::type_list::domain_id_types,
                        gridtools::ghex::bindings::python::type_list::dims>;
}

using halo_generator_specializations = gridtools::meta::transform<
    gridtools::meta::rename<gridtools::ghex::structured::regular::halo_generator>::template apply,
    detail::halo_generator_args>;

}
}
}
}
}
}
}
#endif