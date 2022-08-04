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
#ifndef INCLUDED_GHEX_PYBIND_PATTERN_HPP
#define INCLUDED_GHEX_PYBIND_PATTERN_HPP

#include <gridtools/meta.hpp>
#include <ghex/pattern.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/bindings/python/type_list.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {

namespace detail {
    using pattern_container_args = gridtools::meta::cartesian_product<
                        gridtools::ghex::bindings::python::type_list::grid_types,
                        gridtools::meta::list<gridtools::ghex::bindings::python::type_list::transport>,
                        gridtools::ghex::bindings::python::type_list::domain_id_types,
                        gridtools::ghex::bindings::python::type_list::dims>;

    template <typename GridType, typename Transport, typename DomainIdType, typename Dim>
    struct PatternContainerTypeDeductor {};

    template <typename Transport, typename DomainIdType, typename Dim>
    struct PatternContainerTypeDeductor<gridtools::ghex::structured::grid, Transport, DomainIdType, Dim> {
        using grid_type = gridtools::ghex::structured::grid;

        using context_type = typename gridtools::ghex::tl::context_factory<Transport>::context_type;
        // TODO: paramterize
        using halo_generator_type = gridtools::ghex::structured::regular::halo_generator<DomainIdType, Dim>;
        using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<DomainIdType, Dim>;
        using domain_range_type = std::vector<domain_descriptor_type>;

        using field_type_args = gridtools::meta::cartesian_product<
                        gridtools::ghex::bindings::python::type_list::data_types,
                        gridtools::ghex::bindings::python::type_list::architecture_types,
                        gridtools::meta::list<domain_descriptor_type>,
                        gridtools::ghex::bindings::python::type_list::layout_maps>;

        using field_descriptor_types = gridtools::meta::transform<
            gridtools::meta::rename<gridtools::ghex::structured::regular::field_descriptor>::template apply,
            field_type_args>;

        using pattern_container_type = decltype(gridtools::ghex::make_pattern<grid_type>(
            std::declval<context_type&>(),
            std::declval<halo_generator_type&>(),
            std::declval<typename gridtools::ghex::bindings::python::type_list::domain_range_type<domain_descriptor_type>&>()));
    };
}

template <typename GridType, typename Transport, typename DomainIdType, typename Dim>
using pattern_container_type = typename detail::PatternContainerTypeDeductor<
    GridType, Transport, DomainIdType, Dim>::pattern_container_type;

using pattern_container_specializations = gridtools::meta::transform<
    gridtools::meta::rename<pattern_container_type>::template apply, detail::pattern_container_args>;

template <typename PatternContainer>
using pattern_container_trait = gridtools::meta::rename<
    detail::PatternContainerTypeDeductor,
    gridtools::meta::at<detail::pattern_container_args, typename gridtools::meta::find<pattern_container_specializations, PatternContainer>::type>>;

}
}
}
}
}
#endif