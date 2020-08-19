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
#include "./iterator_iface.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<typename Iterator>
struct put_range_iface
{
    virtual ~put_range_iface() {}

    GT_HOST_DEVICE
    virtual Iterator  begin() const noexcept = 0;
    GT_HOST_DEVICE
    virtual Iterator  end()   const noexcept = 0;
    virtual void      start_local_epoch() = 0;
    virtual void      end_local_epoch() = 0;
    virtual void      start_remote_epoch() = 0;
    virtual void      end_remote_epoch() = 0;
    virtual size_type buffer_size() const = 0;
};

// remote range representation
template<typename Range, typename Iterator, typename SourceArch>
struct put_range_impl : public put_range_iface<Iterator>
{
    Range m;

    put_range_impl(Range&& r) noexcept : m{std::move(r)}
    {
        m.init(SourceArch{});
    }

    ~put_range_impl()
    {
        m.exit(SourceArch{});
    }
    
    GT_FUNCTION
    Iterator  begin() const noexcept override final { return m.begin(); }
    GT_FUNCTION
    Iterator  end()   const noexcept override final { return m.end(); }

    void      start_local_epoch() override final { }
    void      end_local_epoch() override final { }
    void      start_remote_epoch() override final { m.start_remote_epoch(SourceArch{}); }
    void      end_remote_epoch() override final { m.end_remote_epoch(SourceArch{}); }
    size_type buffer_size() const override final { return m.buffer_size(); }
};

// local range representation
template<typename Range, typename Iterator>
struct put_range_impl<Range, Iterator, target_> : public put_range_iface<Iterator>
{
    Range m;

    put_range_impl(Range&& r) : m{std::move(r)} { }

    GT_FUNCTION
    Iterator  begin() const noexcept override final { return m.begin(); }
    GT_FUNCTION
    Iterator  end()   const noexcept override final { return m.end(); }
    void      start_local_epoch() override final { m.start_local_epoch(); }
    void      end_local_epoch() override final { m.end_local_epoch(); }
    void      start_remote_epoch() override final {}
    void      end_remote_epoch() override final {}
    size_type buffer_size() const override final { return m.buffer_size(); }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_IFACE_HPP */
