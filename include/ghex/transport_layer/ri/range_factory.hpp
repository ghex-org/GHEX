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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_FACTORY_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_FACTORY_HPP

#include <vector>
#include <boost/mp11.hpp>
#include "./range.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<typename RangeList>
struct range_factory
{
    static_assert(boost::mp11::mp_is_set<RangeList>::value, "range types must be unique");

    template<typename Range>
    using iterator_p = typename Range::iterator;
    using iterator_list = boost::mp11::mp_transform<iterator_p, RangeList>;
    template<typename Iterator>
    using iterator_size_p = boost::mp11::mp_size_t<sizeof(iterator_impl<Iterator>)>;
    using max_iterator_size =
        boost::mp11::mp_max_element<boost::mp11::mp_transform<iterator_size_p, iterator_list>, boost::mp11::mp_less>;
    using stack_iterator = iterator<max_iterator_size::value>;

    template<typename Range>
    using range_size_p = boost::mp11::mp_size_t<sizeof(put_range_impl<Range, stack_iterator, target_>)>;
    using max_range_size =
        boost::mp11::mp_max_element<boost::mp11::mp_transform<range_size_p, RangeList>, boost::mp11::mp_less>;
    using range_type = put_range<max_range_size::value, max_iterator_size::value>;

    template<typename Range, typename Arch>
    static range_type create(Arch a, Range&& r)
    {
        using range_t = std::remove_cv_t<std::remove_reference_t<Range>>;
        static_assert(boost::mp11::mp_set_contains<RangeList, range_t>::value, "range type not registered");
        using id = boost::mp11::mp_find<RangeList, range_t>;
        return {std::forward<Range>(r), a, id::value};
    }

    template<typename Range, typename... Args>
    static range_type create(Args&&... args)
    {
        return create(target, Range(std::forward<Args>(args)...));
    }

    template<typename Arch>
    static range_type deserialize(Arch a, const byte* data)
    {
        int id;
        std::memcpy(&id, data, sizeof(int));
        data += sizeof(int);
        return boost::mp11::mp_with_index<boost::mp11::mp_size<RangeList>::value>(id, [data, a](auto Id) {
            using range_t = boost::mp11::mp_at<RangeList, decltype(Id)>;
            return create(a, std::move(reinterpret_cast<put_range_impl<range_t, stack_iterator, target_>*>(const_cast<byte*>(data))->m));//.init();
        });
    }

    static constexpr std::size_t serial_size = sizeof(int) + max_range_size::value;

    static void serialize(const range_type& r, byte* buffer)
    {
        byte* data = buffer;
        std::memcpy(data, &r.m_id, sizeof(int));
        data += sizeof(int);
        std::memcpy(data, &r.m_stack, max_range_size::value);
    }

    static std::vector<byte> serialize(const range_type& r)
    {
        std::vector<byte> res(sizeof(int) + max_range_size::value);
        serialize(r, res.data());
        return res;
    }

    static bool on_gpu(const range_type& r)
    {
        return boost::mp11::mp_with_index<boost::mp11::mp_size<RangeList>::value>(r.m_id, [](auto Id) {
            using range_t = boost::mp11::mp_at<RangeList, decltype(Id)>;
            using arch_t = typename range_t::arch_type;
            return std::is_same<gridtools::ghex::gpu, arch_t>::value;
        });
    }

    template<typename Arch, typename SourceRange>
    static void put(Arch, SourceRange& sr, range_type& tr)
    {
        boost::mp11::mp_with_index<boost::mp11::mp_size<RangeList>::value>(tr.m_id, [&sr,&tr](auto Id) {
            using range_t = boost::mp11::mp_at<RangeList, decltype(Id)>;
            sr.put(
                dynamic_cast<put_range_impl<range_t, stack_iterator, Arch>&>(tr.iface()).m
            );
        });
    }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_FACTORY_HPP */
