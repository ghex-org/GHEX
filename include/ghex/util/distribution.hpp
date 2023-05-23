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

#include <array>
#include <cassert>

namespace ghex
{
/** @brief Stores the extents of an N-dimensional hypercube
  * Given a scalar index it computes the coordinates within this cube
  * @tparam N dimension*/
template<unsigned int N>
class dims_map
{
  public:
    using size_type = int;
    using array_type = std::array<size_type, N>;

  private:
    bool       m_reverse = false;
    array_type m_dims;
    size_type  m_size;
    array_type m_partial_product;

  public:
    /** @brief Default constructor: cube has unit extents */
    dims_map() noexcept
    {
        m_size = 1;
        for (unsigned int i = 0; i < N; ++i)
        {
            m_dims[i] = 1;
            m_partial_product[i] = 1;
        }
    }

    /** @brief Construct from extents
      * @param dims_ extents of the hypercube
      * @param reverse indicates the stride: if true, the last dimension changes fastest. */
    dims_map(const array_type& dims_, bool reverse) noexcept
    : m_reverse{reverse}
    , m_dims{dims_}
    {
        m_size = 1;
        for (auto s : m_dims) m_size *= s;
        if (reverse)
        {
            m_partial_product[N - 1] = 1;
            for (unsigned int i = N - 1; i > 0; --i)
                m_partial_product[i - 1] = m_dims[i] * m_partial_product[i];
        }
        else
        {
            m_partial_product[0] = 1;
            for (unsigned int i = 1; i < N; ++i)
                m_partial_product[i] = m_dims[i - 1] * m_partial_product[i - 1];
        }
    }

    dims_map(const dims_map&) = default;
    dims_map(dims_map&&) = default;

    dims_map& operator=(const dims_map&) = default;
    dims_map& operator=(dims_map&&) = default;

  public:
    size_type         size() const noexcept { return m_size; }
    const array_type& dims() const noexcept { return m_dims; }

  public:
    /** @brief Computes the coordinate within the hypercube
      * @param idx Scalar index (enumerator)
      * @return Coordinates according to the given index */
    array_type operator()(size_type idx) const noexcept
    {
        assert(idx < m_size);
        array_type res;
        if (m_reverse)
            for (unsigned int i = 0; i < N; ++i)
            {
                res[i] = idx / m_partial_product[i];
                idx -= res[i] * m_partial_product[i];
            }
        else
            for (unsigned int i = N; i > 0; --i)
            {
                res[i - 1] = idx / m_partial_product[i - 1];
                idx -= res[i - 1] * m_partial_product[i - 1];
            }
        return res;
    }
};

// alias definition used for a hierarchical, regular domain decomposition
template<unsigned int Levels>
using hierarchical_distribution = dims_map<Levels>;

} // namespace ghex
