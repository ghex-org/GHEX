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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_IFACE_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_IFACE_HPP

#include <cstring>
#include <gridtools/common/host_device.hpp>
#include "./types.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

struct iterator_iface
{
    virtual ~iterator_iface() {}

    GT_HOST_DEVICE
    virtual chunk operator*() const noexcept = 0;

    GT_HOST_DEVICE
    operator chunk() const noexcept { return this->operator*(); }

    GT_HOST_DEVICE
    virtual iterator_iface& operator++()          noexcept = 0;
    GT_HOST_DEVICE
    virtual iterator_iface& operator--()          noexcept = 0;
    GT_HOST_DEVICE
    virtual iterator_iface& operator+=(size_type) noexcept = 0;

    GT_HOST_DEVICE
    virtual size_type sub(const iterator_iface&) const noexcept = 0;

    GT_HOST_DEVICE
    virtual bool equal(const iterator_iface&) const noexcept = 0;
    GT_HOST_DEVICE
    virtual bool lt(const iterator_iface&) const noexcept = 0;
};

template<typename Iterator>
struct iterator_impl : public iterator_iface
{
    Iterator m;
    GT_FUNCTION
    iterator_impl(const Iterator& it) noexcept : m{it} {}

    GT_FUNCTION
    chunk operator*() const noexcept override final { return *m; }

    GT_FUNCTION
    iterator_impl& operator++()            noexcept override final { ++m; return *this; }
    GT_FUNCTION
    iterator_impl& operator--()            noexcept override final { --m; return *this; }
    GT_FUNCTION
    iterator_impl& operator+=(size_type n) noexcept override final { m+=n; return *this; }

    GT_FUNCTION
    size_type sub(const iterator_iface& other) const noexcept override final {
        return m.sub(static_cast<const iterator_impl&>(other).m);
    }

    GT_FUNCTION
    bool equal(const iterator_iface& other) const noexcept override final {
        return m.equal(static_cast<const iterator_impl&>(other).m);
    }
    GT_FUNCTION
    bool lt(const iterator_iface& other) const noexcept override final {
        return m.lt(static_cast<const iterator_impl&>(other).m);
    }
};


} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_ITERATOR_IFACE_HPP */
