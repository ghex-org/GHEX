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
#ifndef INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP
#define INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP

#include <gridtools/common/host_device.hpp>
#include "../rma/chunk.hpp"

namespace gridtools {
namespace ghex {
namespace structured {

/** @brief A simple iterator which works on multi-dimensional ranges. Furthermore, this type can be
  * used on the device to access the range.
  * @tparam Range The underlying range type */
template<typename Range>
struct range_iterator
{
    using coordinate = typename Range::coordinate;
    using chunk = rma::chunk<typename Range::value_type>;
    using size_type = unsigned int;

    Range*      m_range;
    size_type   m_index;
    coordinate  m_coord;

    GT_FUNCTION
    range_iterator(Range* r, size_type idx, const coordinate& coord)
    : m_range{r}
    , m_index{idx}
    , m_coord{coord}
    {}
    GT_FUNCTION
    range_iterator(const range_iterator& other) noexcept
    : m_range{other.m_range}
    , m_index{other.m_index}
    , m_coord{other.m_coord}
    {}
    GT_FUNCTION
    range_iterator(range_iterator&& other) noexcept
    : m_range{other.m_range}
    , m_index{other.m_index}
    , m_coord{other.m_coord}
    {}

    GT_HOST_DEVICE
    auto index() const noexcept { return m_index; }
    
    GT_HOST_DEVICE
    chunk     operator*() const noexcept { return m_range->get_chunk(m_coord); }
    GT_HOST_DEVICE
    void      operator++() noexcept { m_index = m_range->inc(m_index, m_coord); }
    GT_HOST_DEVICE
    void      operator--() noexcept { m_range->inc(m_index, -1, m_coord); }
    GT_HOST_DEVICE
    void      operator+=(int n) noexcept { m_range->inc(m_index, n, m_coord); }
    GT_FUNCTION
    size_type sub(const range_iterator& other) const { return m_index - other.m_index; }
    GT_FUNCTION
    bool      equal(const range_iterator& other) const { return m_index == other.m_index; }
    GT_FUNCTION
    bool      lt(const range_iterator& other) const { return m_index < other.m_index; }
};

} // namespace structured
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_RMA_RANGE_ITERATOR_HPP */
