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

#include <ghex/rma/locality.hpp>
#include <ghex/rma/handle.hpp>
#include <ghex/rma/access_guard.hpp>
#include <ghex/rma/event.hpp>
#include <ghex/rma/range_iface.hpp>
#include <memory>

namespace ghex
{
namespace rma
{
/** @brief generic range type which is created through de-serialization of a target range. Hence,
  * this range is created at the source side of a put operation and represents the remot target of
  * this operation. It exposes member functions to get read/write access to the remote resource.
  * The (void) pointer to the shared memory resource can be obtained. In order to be useful, this
  * type can be converted to its underlying concrete type through type injection using the
  * range_factory. */
struct range
{
    int                          m_id = 0;
    locality                     m_loc;
    remote_handle                m_handle;
    std::unique_ptr<range_iface> m_impl;
    remote_access_guard          m_guard;
    bool                         m_on_gpu;
    remote_event                 m_event;

    range() = default;

    template<typename Range>
    range(Range&& r, int id, info field_info, typename local_access_guard::info info_,
        event_info e_info_, int rank, bool on_gpu_)
    : m_id{id}
    , m_loc{info_.m_locality}
    , m_impl{new range_impl<std::remove_reference_t<Range>>(std::forward<Range>(r))}
    , m_guard(info_, rank)
    , m_on_gpu{on_gpu_}
    , m_event{e_info_, m_on_gpu}
    {
        m_handle.init(field_info, m_loc, rank);
    }

    range(range&&) = default;

    range& operator=(range&&) = default;

    void* get_ptr() const { return m_handle.get_ptr(m_loc); }

    void start_source_epoch() { m_guard.start_source_epoch(); }
    bool try_start_source_epoch() { return m_guard.try_start_source_epoch(); }
    void end_source_epoch() { m_guard.end_source_epoch(); }
};

} // namespace rma
} // namespace ghex
