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
#ifndef INCLUDED_GHEX_RMA_RANGE_FACTORY_HPP
#define INCLUDED_GHEX_RMA_RANGE_FACTORY_HPP

#include <vector>
#include <cstring>
#include <boost/mp11.hpp>
#include "./range.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

template<typename RangeList>
struct range_factory
{
    using range_type = range;

    static_assert(boost::mp11::mp_is_set<RangeList>::value, "range types must be unique");
    
    template<typename Range>
    using range_size_p = boost::mp11::mp_size_t<sizeof(Range)>;
    using max_range_size =
        boost::mp11::mp_max_element<boost::mp11::mp_transform<range_size_p, RangeList>, boost::mp11::mp_less>;
    
    static constexpr std::size_t serial_size = sizeof(int)*2 + max_range_size::value;

    template<typename Range>
    static void serialize(const Range& r, unsigned char* buffer)
    {
        static_assert(boost::mp11::mp_set_contains<RangeList, Range>::value, "range type not registered");
        using id = boost::mp11::mp_find<RangeList, Range>;
        const int m_id = id::value;
        std::memcpy(buffer, &m_id, sizeof(int));
        buffer += 2*sizeof(int);
        std::memcpy(buffer, &r, sizeof(Range));
    }

    template<typename Range>
    static std::vector<unsigned char> serialize(const Range& r)
    {
        std::vector<unsigned char> res(serial_size);
        serialize(r, res.data());
        return res;
    }

    static range deserialize(unsigned char* buffer)
    {
        int id;
        std::memcpy(&id, buffer, sizeof(int));
        buffer += 2*sizeof(int);
        return boost::mp11::mp_with_index<boost::mp11::mp_size<RangeList>::value>(id, [buffer](auto Id)
        {
            using range_t = boost::mp11::mp_at<RangeList, decltype(Id)>;
            return range(std::move(*reinterpret_cast<range_t*>(buffer)), decltype(Id)::value);
        });
    }

    // calls sr.put(_tr) where _tr is the recovered concrete type of the target range
    // (i.e. no longer type erased)
    template<typename SourceRange>
    static void put(SourceRange& sr, range& tr)
    {
        boost::mp11::mp_with_index<boost::mp11::mp_size<RangeList>::value>(tr.m_id, [&sr,&tr](auto Id)
        {
            using range_t = boost::mp11::mp_at<RangeList, decltype(Id)>;
            sr.put(dynamic_cast<range_impl<range_t>*>(tr.m_impl.get())->m);
        });
    }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_FACTORY_HPP */
