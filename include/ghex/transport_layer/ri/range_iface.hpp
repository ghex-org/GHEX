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

template<typename Iterator, typename SourceRange>
struct range_iface
{
    virtual ~range_iface() {}

    virtual Iterator  begin() const noexcept = 0;
    virtual Iterator  end()   const noexcept = 0;
    virtual void      put(const chunk&, const byte*) = 0;
    virtual void put(const SourceRange&) {};
    //virtual Iterator      put(Iterator&, const byte*) = 0;
    virtual void      start_local_epoch() = 0;
    virtual void      end_local_epoch() = 0;
    virtual void      start_remote_epoch() = 0;
    virtual void      end_remote_epoch() = 0;
    virtual size_type buffer_size() const = 0;
};

template<typename Range, typename Iterator, typename TargetRange, typename Arch>
struct range_impl : public range_iface<Iterator, TargetRange>
{
    Range m;

    range_impl(Range&& r) noexcept : m{std::move(r)}
    {
        m.init(Arch{});
    }

    ~range_impl()
    {
        m.exit(Arch{});
    }
    
    Iterator  begin() const noexcept override final { return m.begin(); }
    Iterator  end()   const noexcept override final { return m.end(); }
    void      put(const chunk& c, const byte* ptr) override final 
    { 
        Range::put(c, ptr, Arch{});
    }
    virtual void put(const TargetRange& r) override final
    {
        m.put(r, Arch{});
    };
    /*Iterator     put(Iterator& it, const byte* ptr) override final{

        //Range::put(*it,ptr,Arch{});
        //return it;
        
        using iterator_iface_t = std::remove_reference_t<decltype(it.iface())>;
        using concrete_iterator_t = typename Range::iterator;
        using iterator_impl_t = iterator_impl<concrete_iterator_t>;
        ////Range::put(it, ptr, Arch{}); 
        iterator_iface_t& iface_it = it.iface();
        iterator_impl_t& impl_it = static_cast<iterator_impl_t&>(iface_it);
        concrete_iterator_t& c_it = impl_it.m;
        return {Range::put(c_it, ptr, Arch{})}; 
        //return it;
    }*/
    void      start_local_epoch() override final { }
    void      end_local_epoch() override final { }
    void      start_remote_epoch() override final { m.start_remote_epoch(Arch{}); }
    void      end_remote_epoch() override final { m.end_remote_epoch(Arch{}); }
    size_type buffer_size() const override final { return m.buffer_size(); }
};

template<typename Range, typename Iterator, typename SourceRange>
struct range_impl<Range, Iterator, SourceRange, target_> : public range_iface<Iterator, SourceRange>
{
    Range m;

    range_impl(Range&& r) : m{std::move(r)} { }

    Iterator  begin() const noexcept override final { return m.begin(); }
    Iterator  end()   const noexcept override final { return m.end(); }
    void      put(const chunk&, const byte*) override final { }
    //Iterator      put(Iterator& it, const byte*) override final { return it; }
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
