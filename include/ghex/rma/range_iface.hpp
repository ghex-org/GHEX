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

namespace gridtools {
namespace ghex {
namespace rma {

// type erasure mechanism for ranges
struct range_iface
{
    virtual ~range_iface() {}
};

template<typename Range>
struct range_impl : public range_iface
{
    Range m;

    range_impl(Range&& r) noexcept
    : m{std::move(r)}
    {}
    
    range_impl(range_impl&&) = default;

    ~range_impl()
    {}
};

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_RANGE_IFACE_HPP */
