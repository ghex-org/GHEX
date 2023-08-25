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

namespace ghex
{
namespace rma
{
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
    {
    }

    range_impl(range_impl&&) = default;

    ~range_impl() {}
};

} // namespace rma
} // namespace ghex
