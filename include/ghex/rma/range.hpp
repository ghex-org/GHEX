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
#ifndef INCLUDED_GHEX_RMA_RANGE_HPP
#define INCLUDED_GHEX_RMA_RANGE_HPP

#include <memory>
#include "./locality.hpp"
#include "./handle.hpp"
#include "./range_iface.hpp"
//#include "./access_guard.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

struct range
{
    int m_id = 0;
    locality m_loc;
    remote_handle m_handle;
    std::unique_ptr<range_iface> m_impl;
    
    range() = default;

    template<typename Range>
    range(Range&& r, int id, info field_info, locality loc)
    : m_id{id}
    , m_loc{loc}
    , m_impl{new range_impl<std::remove_reference_t<Range>>(std::forward<Range>(r))}
    {
        m_handle.init(field_info);
    }

    range(range&&) = default;
    
    range& operator=(range&&) = default;

    void* get_ptr() const
    {
        return m_handle.get_ptr(m_loc);
    }

    void start_source_epoch() { m_impl->start_source_epoch(); }
    void end_source_epoch() { m_impl->end_source_epoch(); }
    void start_target_epoch() { m_impl->start_target_epoch(); }
    void end_target_epoch() { m_impl->end_target_epoch(); }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_HPP */

