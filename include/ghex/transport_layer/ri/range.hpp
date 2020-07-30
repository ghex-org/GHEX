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
#ifndef INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP
#define INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP

#include "./range_iface.hpp"
#include "./iterator.hpp"

namespace gridtools {
namespace ghex {
namespace tl {
namespace ri {

template<unsigned int StackMemory, unsigned int IteratorStackMemory>
struct range
{
    using iterator_type = iterator<IteratorStackMemory>;

    int  m_id = 0;
    byte m_stack[StackMemory];

    template<typename Range, typename Arch>
    range(Range&& r, Arch, int id = 0) {
        using range_t = std::remove_cv_t<std::remove_reference_t<Range>>;
        new (m_stack) range_impl<range_t, iterator_type, Arch>{std::forward<Range>(r)};
        m_id = id;
    }
    range() = default;
    range(range&&) = default;
    range& operator=(range&&) = default;

    iterator_type begin() const noexcept { return ciface().begin(); }
    iterator_type end() const noexcept { return ciface().end(); }
    chunk operator[](size_type i) const noexcept { return *(begin() + i); }
    size_type size() const noexcept { return end() - begin(); }

    size_type buffer_size() const { return ciface().buffer_size(); }

    void start_target_epoch() { iface().start_local_epoch(); }
    void end_target_epoch()   { iface().end_local_epoch(); }
    void start_source_epoch() { iface().start_remote_epoch(); }
    void end_source_epoch()   { iface().end_remote_epoch(); }

          range_iface<iterator_type>&  iface()       { return *reinterpret_cast<range_iface<iterator_type>*>(m_stack); }
    const range_iface<iterator_type>&  iface() const { return *reinterpret_cast<const range_iface<iterator_type>*>(m_stack); }
    const range_iface<iterator_type>& ciface() const { return *reinterpret_cast<const range_iface<iterator_type>*>(m_stack); }

    //void put(const chunk& c, const byte* ptr) {} //{ iface().put(c, ptr); }
    void put(const iterator_type& it, const byte* ptr) {
        //chunk c = it.iface();  
        iface().put(it, ptr); 
    }
};

} // namespace ri
} // namespace tl
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TRANSPORT_LAYER_RI_RANGE_HPP */
