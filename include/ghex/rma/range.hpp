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
#include "./access_guard.hpp"
#include "./range_iface.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

/** @brief generic range type which is created through de-serialization of a target range. Hence,
  * this range is created at the source side of a put operation and represents the remot target of
  * this operation. It exposes member functions to get read/write access to the remote resource.
  * The (void) pointer to the shared memory resource can be obtained. In order to be useful, this
  * type can be converted to its underlying concrete type through type injection using the
  * range_factory. */
struct range
{
    int m_id = 0;
    locality m_loc;
    remote_handle m_handle;
    std::unique_ptr<range_iface> m_impl;
    remote_access_guard m_guard;
    
    range() = default;

    template<typename Range>
    range(Range&& r, int id, info field_info, typename local_access_guard::info info_)
    : m_id{id}
    , m_loc{info_.m_locality}
    , m_impl{new range_impl<std::remove_reference_t<Range>>(std::forward<Range>(r))}
    , m_guard(info_)
    {
        m_handle.init(field_info);
    }

    range(range&&) = default;
    
    range& operator=(range&&) = default;

    void* get_ptr() const
    {
        return m_handle.get_ptr(m_loc);
    }

    void start_source_epoch() { m_guard.start_source_epoch(); }
    void end_source_epoch() { m_guard.end_source_epoch(); }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_HPP */

