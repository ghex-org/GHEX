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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_IFACE_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_IFACE_HPP

#include "./types.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<typename Iterator>
struct range_iface
{
    virtual ~range_iface() {}

    virtual Iterator  begin() const noexcept = 0;
    virtual Iterator  end()   const noexcept = 0;
    virtual void      put(const chunk&, const byte*) = 0;
    virtual void      start_local_epoch() = 0;
    virtual void      end_local_epoch() = 0;
    virtual void      start_remote_epoch() = 0;
    virtual void      end_remote_epoch() = 0;
    virtual void      init() = 0;
    virtual size_type buffer_size() const = 0;
};

template<typename Range, typename Iterator, typename Arch>
struct range_impl : public range_iface<Iterator>
{
    Range m;

    range_impl(Range&& r) noexcept : m{std::move(r)} { }
    
    Iterator  begin() const noexcept override final { return m.begin(); }
    Iterator  end()   const noexcept override final { return m.end(); }
    void      put(const chunk& c, const byte* ptr) override final { Range::put(c, ptr, Arch{}); }
    void      start_local_epoch() override final { }
    void      end_local_epoch() override final { }
    void      start_remote_epoch() override final { m.start_remote_epoch(Arch{}); }
    void      end_remote_epoch() override final { m.end_remote_epoch(Arch{}); }
    void      init() override final { m.init(Arch{}); }
    size_type buffer_size() const override final { return m.buffer_size(); }
};

template<typename Range, typename Iterator>
struct range_impl<Range, Iterator, target_> : public range_iface<Iterator>
{
    Range m;

    range_impl(Range&& r) : m{std::move(r)} { }

    Iterator  begin() const noexcept override final { return m.begin(); }
    Iterator  end()   const noexcept override final { return m.end(); }
    void      put(const chunk&, const byte*) override final { }
    void      start_local_epoch() override final { m.start_local_epoch(); }
    void      end_local_epoch() override final { m.end_local_epoch(); }
    void      start_remote_epoch() override final {}
    void      end_remote_epoch() override final {}
    void      init() override final {}
    size_type buffer_size() const override final { return m.buffer_size(); }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_IFACE_HPP */
