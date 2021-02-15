/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_GLUE_GRIDTOOLS_FIELD_HPP
#define INCLUDED_GHEX_GLUE_GRIDTOOLS_FIELD_HPP

#include "../../structured/regular/field_descriptor.hpp"
#include "../../arch_traits.hpp"
#include "./processor_grid.hpp"
#include <gridtools/storage/data_store.hpp>
#include <gridtools/meta/list_to_iseq.hpp>

namespace gridtools {


    namespace ghex {
        namespace _impl {

            template<typename Halo, typename S, typename I, I... Is>
            array<S, sizeof...(Is)>
            get_begin(std::integer_sequence<I,Is...>)
            {
                return {Halo::template at<Is>()...};
            }

            template<typename Seq>
            struct get_layout_map;

            template<template <class T, T...> class Seq, typename I, I... Is>
            struct get_layout_map<Seq<I, Is...>>
            {
                using type = ::gridtools::layout_map<Is...>;
            };

            template<typename Layout>
            struct get_unmasked_layout_map;

            template<int... Args>
            struct get_unmasked_layout_map<layout_map<Args...>>
            {
                using args          = ::gridtools::meta::list<std::integral_constant<int, Args>...>;
                using unmasked_args = ::gridtools::meta::filter<::gridtools::_impl::_layout_map::not_negative, args>;
                using integer_seq   = ::gridtools::meta::list_to_iseq<unmasked_args>;
                using type          = typename get_layout_map<integer_seq>::type;
            };

            template<typename T, typename Arch, typename DomainDescriptor, typename Seq>
            struct get_field_descriptor_type;

            template<typename T, typename Arch, typename DomainDescriptor, template <class J, J...> class Seq, typename I, I... Is>
            struct get_field_descriptor_type<T,Arch,DomainDescriptor, Seq<I, Is...>>
            {
                using type = structured::regular::field_descriptor<T,Arch,DomainDescriptor,Is...>;
            };
        } // namespace _impl

        template <typename Arch, typename DomainDescriptor, typename DataStore>
        auto wrap_gt_field(const DomainDescriptor& dom,
                           const std::shared_ptr<DataStore>& ds,
                           const std::array<int, DataStore::ndims>& origin,
                           int device_id = 0)
        {
            using value_t           = typename DataStore::data_t;
            using layout_t          = typename DataStore::layout_t;
            using integer_seq       = typename _impl::get_unmasked_layout_map<layout_t>::integer_seq;
            using uint_t            = decltype(layout_t::masked_length);
            using dimension         = std::integral_constant<uint_t, layout_t::masked_length>;
            using field_desc_t      = typename _impl::get_field_descriptor_type<
                value_t, Arch, DomainDescriptor, integer_seq>::type;

            auto strides = ds->strides();
            for (unsigned int i=0u; i<dimension::value; ++i)
                strides[i] *= sizeof(value_t);

            return field_desc_t(dom, ds->get_target_ptr(), origin, ds->lengths(), strides, 1, false, device_id);
        }

        template <typename Arch, typename Transport, typename DataStore, typename Origin>
        auto wrap_gt_field(const gt_grid<Transport>& grid, DataStore&& ds, Origin&& origin, int device_id = 0)
        {
            return wrap_gt_field<Arch>(grid.m_domains[0], std::forward<DataStore>(ds), std::forward<Origin>(origin), device_id);
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_GLUE_GRIDTOOLS_FIELD_HPP */

