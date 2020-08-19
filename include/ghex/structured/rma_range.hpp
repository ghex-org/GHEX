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

#include "../rma/locality.hpp"
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
    using size_type = unsigned int;
    using iterator = range_iterator<rma_range>;

    guard_view_type   m_guard;
    view_type         m_view;
    
    rma_range(const view_type& v, guard_type& g, rma::locality loc) noexcept
    : m_guard{g, loc}
    , m_view{v}
    {}
    
    rma_range(const rma_range&) = default;
    rma_range(rma_range&&) = default;

    // these functions are called at the remote site upon deserializing and reconstructing the range
    // and can be used to allocate state
    void init()
    {
        m_view.m_field.reset_rma_data();
        m_view.m_field.init_rma_remote(m_view.m_rma_data, m_guard.get_locality());
	    m_guard.init_remote();
    }
    void exit()
    {
        m_view.m_field.release_rma_remote();
        m_guard.release_remote(); 
    }
    
    void start_target_epoch() { m_guard.start_local_epoch(); }
    void end_target_epoch() { m_guard.end_local_epoch(); }

    void start_source_epoch() { m_guard.start_remote_epoch(); }
    void end_source_epoch() { m_guard.end_remote_epoch(); }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_HPP */
