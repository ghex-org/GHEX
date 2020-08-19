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
#include "./range_iface.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

struct range
{
    int m_id = 0;
    std::unique_ptr<range_iface> m_impl;
    
    range() = default;

    template<typename Range>
    range(Range&& r, int id)
    : m_id{id}
    , m_impl{new range_impl<std::remove_reference_t<Range>>(std::forward<Range>(r))}
    {}

    range(range&&) = default;
    
    range& operator=(range&&) = default;

    void start_source_epoch() { m_impl->start_source_epoch(); }
    void end_source_epoch() { m_impl->end_source_epoch(); }
    void start_target_epoch() { m_impl->start_target_epoch(); }
    void end_target_epoch() { m_impl->end_target_epoch(); }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_HPP */

