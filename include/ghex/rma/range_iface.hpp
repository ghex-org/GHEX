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
#ifndef INCLUDED_GHEX_RMA_RANGE_IFACE_HPP
#define INCLUDED_GHEX_RMA_RANGE_IFACE_HPP

#include <gridtools/common/host_device.hpp>
#include "../common/moved_bit.hpp"

namespace gridtools {
namespace ghex {
namespace rma {

struct range_iface
{
    virtual ~range_iface() {}
    virtual void start_source_epoch() = 0;
    virtual void end_source_epoch() = 0;
    virtual void start_target_epoch() = 0;
    virtual void end_target_epoch() = 0;
};

template<typename Range>
struct range_impl : public range_iface
{
    Range m;
    moved_bit m_moved;

    range_impl(Range&& r) noexcept
    : m{std::move(r)}
    {
        m.init();
    }
    
    range_impl(range_impl&&) = default;

    ~range_impl()
    {
        if (!m_moved) m.exit();
    }
    
    void start_source_epoch() override final { m.start_source_epoch(); }
    void end_source_epoch() override final { m.end_source_epoch(); }
    void start_target_epoch() override final { m.start_target_epoch(); }
    void end_target_epoch() override final { m.end_target_epoch(); }
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_IFACE_HPP */
