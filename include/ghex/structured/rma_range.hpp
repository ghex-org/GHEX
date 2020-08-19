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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP

#include "./rma_field_view.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

template<typename Field>
struct rma_range
{
    using arch_type = typename Field::arch_type;
    using view_type = field_view<Field>;
    using layout = typename Field::layout_map;
    using dimension = typename Field::dimension;
    using value_type = typename Field::value_type;
    using coordinate = typename Field::coordinate_type;
    using strides_type = typename Field::strides_type;
    using guard_type = typename view_type::guard_type;
    using guard_view_type = typename view_type::guard_view_type;
    using size_type = tl::ri::size_type;
    using iterator = range_iterator<rma_range>;

    guard_view_type   m_guard;
    view_type         m_view;
    
    rma_range(const view_type& v, guard_type& g, tl::ri::locality loc) noexcept
    : m_guard{g, loc}
    , m_view{v}
    {}
    
    rma_range(const rma_range&) = default;
    rma_range(rma_range&&) = default;

    GT_FUNCTION
    iterator  begin() const { return {const_cast<rma_range*>(this), 0, m_view.m_begin}; }
    GT_FUNCTION
    iterator  end()   const { return {const_cast<rma_range*>(this), m_view.m_size, m_view.m_end}; }
    GT_FUNCTION
    size_type buffer_size() const { return m_view.m_chunk_size; }

    // these functions are called at the remote site upon deserializing and reconstructing the range
    // and can be used to allocate state
    void init(tl::ri::remote_host_)   
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data, m_guard.get_locality());
	    m_guard.init_remote();
    }
    void init(tl::ri::remote_device_)
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data, m_guard.get_locality());
	    m_guard.init_remote();
    }
    void exit(tl::ri::remote_host_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    void exit(tl::ri::remote_device_)
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    
    void start_local_epoch() { m_guard.start_local_epoch(); }
    void end_local_epoch()   { m_guard.end_local_epoch(); }

    void start_remote_epoch(tl::ri::remote_host_)   { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_host_)     { m_guard.end_remote_epoch(); }
    void start_remote_epoch(tl::ri::remote_device_) { m_guard.start_remote_epoch(); }
    void end_remote_epoch(tl::ri::remote_device_)   { m_guard.end_remote_epoch(); }
    
    GT_FUNCTION
    auto get_chunk(const coordinate& coord) const noexcept {
        return m_view.get_chunk(coord);
    }

    GT_FUNCTION
    void inc(size_type& index, size_type n, coordinate& coord) const noexcept {
        m_view.inc(index,n,coord);
    }
    
    GT_FUNCTION
    size_type inc(size_type index, coordinate& coord) const noexcept {
        return m_view.inc(index, coord);
    }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP */
