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
#ifndef INCLUDED_GHEX_PYBIND_CONTEXT_HPP
#define INCLUDED_GHEX_PYBIND_CONTEXT_HPP

#include <gridtools/meta.hpp>
#include <ghex/transport_layer/context.hpp>
#include <ghex/bindings/python/type_list.hpp>
#include <ghex/transport_layer/mpi/context.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {
namespace transport_layer {

namespace detail {
    using context_args = gridtools::meta::cartesian_product<
                        gridtools::meta::list<gridtools::ghex::tl::mpi_tag>>;

    template<typename Transport>
    using context_type = typename gridtools::ghex::tl::context_factory<Transport>::context_type;
}

using context_specializations = gridtools::meta::transform<
    gridtools::meta::rename<detail::context_type>::template apply,
    detail::context_args>;

}
}
}
}
}
}
#endif