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
#ifndef INCLUDED_GHEX_PYBIND_BUFFER_INFO_HPP
#define INCLUDED_GHEX_PYBIND_BUFFER_INFO_HPP

#include <gridtools/meta.hpp>
#include <ghex/bindings/python/type_list.hpp>
#include <ghex/bindings/python/types/pattern.hpp>
#include <ghex/buffer_info.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {

namespace detail {
    template <typename pattern_container_type, typename field_descriptor_type>
    using buffer_info_type = gridtools::ghex::buffer_info<
        typename pattern_container_type::value_type,
        typename field_descriptor_type::arch_type,
        field_descriptor_type>;
}

template <typename pattern_container_type>
using buffer_info_specializations_per_pattern_container = gridtools::meta::transform<
    gridtools::meta::bind<detail::buffer_info_type, pattern_container_type, gridtools::meta::_1>::template apply,
    typename pattern_container_trait<pattern_container_type>::field_descriptor_types>;

using buffer_info_specializations = gridtools::meta::rename<gridtools::meta::concat, gridtools::meta::transform<
    buffer_info_specializations_per_pattern_container, pattern_container_specializations>>;

}
}
}
}
}
#endif