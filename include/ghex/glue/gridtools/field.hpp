/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <type_traits>

#include <ghex/config.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/glue/gridtools/processor_grid.hpp>
#include <gridtools/storage/data_store.hpp>
#include <gridtools/meta/list_to_iseq.hpp>

namespace ghex
{
namespace _impl
{
//template<typename Halo, typename S, typename I, I... Is>
//array<S, sizeof...(Is)> get_begin(std::integer_sequence<I, Is...>)
//{
//    return {Halo::template at<Is>()...};
//}

template <class Int>
using not_negative = std::integral_constant<bool, (Int::value >= 0)>;

template<typename Seq>
struct get_layout_map;

template<template<class T, T...> class Seq, typename I, I... Is>
struct get_layout_map<Seq<I, Is...>>
{
    using type = gridtools::layout_map<Is...>;
};

template<typename Layout>
struct get_unmasked_layout_map;

template<int... Args>
struct get_unmasked_layout_map<gridtools::layout_map<Args...>>
{
    using args = gridtools::meta::list<std::integral_constant<int, Args>...>;
    using unmasked_args =
        gridtools::meta::filter<not_negative, args>;
    using integer_seq = gridtools::meta::list_to_iseq<unmasked_args>;
    using type = typename get_layout_map<integer_seq>::type;
};

} // namespace _impl

template<typename Arch, typename DomainDescriptor, typename DataStore>
auto
wrap_gt_field(const DomainDescriptor& dom, const std::shared_ptr<DataStore>& ds,
    const std::array<int, DataStore::ndims>& origin, int device_id =
    arch_traits<Arch>::current_id())
{
    using value_t = typename DataStore::data_t;
    using layout_t = typename DataStore::layout_t;
    using unmasked_layout_t = typename _impl::get_unmasked_layout_map<layout_t>::type;
    using uint_t = decltype(layout_t::masked_length);
    using dimension = std::integral_constant<uint_t, layout_t::masked_length>;
    using field_desc_t =
        structured::regular::field_descriptor<value_t, Arch, DomainDescriptor, unmasked_layout_t>;

    auto strides = ds->strides();
    for (unsigned int i = 0u; i < dimension::value; ++i) strides[i] *= sizeof(value_t);

    return field_desc_t(
        dom, ds->get_target_ptr(), origin, ds->lengths(), strides, 1, false, device_id);
}

template<typename Arch, typename DataStore, typename Origin>
auto
wrap_gt_field(const gt_grid& grid, DataStore&& ds, Origin&& origin, int device_id = arch_traits<Arch>::current_id())
{
    return wrap_gt_field<Arch>(
        grid.m_domains[0], std::forward<DataStore>(ds), std::forward<Origin>(origin), device_id);
}

} // namespace ghex
