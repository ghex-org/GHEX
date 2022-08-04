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
#ifndef INCLUDED_GHEX_PYBIND_COMMUNICATION_OBJECT_HPP
#define INCLUDED_GHEX_PYBIND_COMMUNICATION_OBJECT_HPP

#include <gridtools/meta.hpp>
#include <ghex/pattern.hpp>
#include <ghex/buffer_info.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/bindings/python/type_list.hpp>
#include <ghex/bindings/python/types/pattern.hpp>
#include <ghex/bindings/python/types/buffer_info.hpp>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {
namespace types {

namespace detail {
    using communication_object_args = gridtools::meta::cartesian_product<
                        gridtools::meta::list<gridtools::ghex::bindings::python::type_list::transport>,
                        gridtools::ghex::bindings::python::type_list::grid_types,
                        gridtools::ghex::bindings::python::type_list::domain_id_types,
                        gridtools::ghex::bindings::python::type_list::dims>;

    template <typename GridType>
    struct DomainDescriptorDeductor {};

    template <>
    struct DomainDescriptorDeductor<gridtools::ghex::structured::grid> {
        template <typename DomainIdType, typename Dim>
        using type = gridtools::ghex::structured::regular::domain_descriptor<DomainIdType, Dim>;
    };

    template <typename GridType, typename DomainIdType, typename Dim>
    using domain_descriptor_type = typename DomainDescriptorDeductor<GridType>::template type<DomainIdType, Dim>;

    template <typename Transport, typename GridType, typename DomainIdType, typename Dim>
    struct CommunicationObjectTypeDeductor {
        using transport = Transport;
        using grid_type = GridType;
        using domain_id_type = DomainIdType;
        using dim = Dim;

        using context_type = typename gridtools::ghex::tl::context_factory<transport>::context_type;
        using communicator_type = typename context_type::communicator_type;
        using grid_impl_type = typename GridType::template type<domain_descriptor_type<GridType, DomainIdType, Dim>>;

        // TODO: could be multiple in the future when the halo generator is parameterized
        using pattern_container_type = typename gridtools::ghex::bindings::python::types::pattern_container_type<
            grid_type, transport, domain_id_type, dim>;

        using buffer_info_types = typename gridtools::ghex::bindings::python::types::buffer_info_specializations_per_pattern_container<pattern_container_type>;

        using communication_object_type = gridtools::ghex::communication_object<
            communicator_type, grid_impl_type, domain_id_type>;

        using communication_handle_type = typename communication_object_type::handle_type;
    };

    template <typename Transport, typename GridType, typename DomainIdType, typename Dim>
    using communication_object_type = typename CommunicationObjectTypeDeductor<
        Transport, GridType, DomainIdType, Dim>::communication_object_type;

    template <typename Transport, typename GridType, typename DomainIdType, typename Dim>
    using communication_handle_type = typename CommunicationObjectTypeDeductor<
        Transport, GridType, DomainIdType, Dim>::communication_handle_type;
}

using communication_object_specializations = gridtools::meta::transform<
    gridtools::meta::rename<detail::communication_object_type>::template apply,
    detail::communication_object_args>;

template <typename CommunicationObject>
using communication_object_trait = gridtools::meta::rename<
    detail::CommunicationObjectTypeDeductor,
    gridtools::meta::at<detail::communication_object_args,
    typename gridtools::meta::find<communication_object_specializations, CommunicationObject>::type>>;

using communication_handle_specializations = gridtools::meta::dedup<gridtools::meta::transform<
    gridtools::meta::rename<detail::communication_handle_type>::template apply,
    detail::communication_object_args>>;

}
}
}
}
}
#endif