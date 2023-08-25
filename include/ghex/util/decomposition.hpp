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

#include <ghex/util/distribution.hpp>
#include <ghex/util/resource_layout.hpp>

namespace ghex
{
/** @brief A class which stores a hierarchical domain decomposition.
  * The hierarchy is established through 4 levels which can be mapped to
  * - nodes per run
  * - numa-nodes per node
  * - ranks per numa-node
  * - threads per rank
  * Each level stores the extents of a D-dimensional hypercube. Given a scalar index
  * or a rank-index, thread-index pair it computes the coordinate in D-dimensional space of the
  * corresponding domain. The relative order within each level is Fortran / first dimension has
  * stride=1.
  * @tparam D dimension */
template<unsigned int D>
class hierarchical_decomposition
{
  private:
    using resource_type = hierarchical_resource_layout<D, 4>;

  public:
    using size_type = typename resource_type::size_type;
    using array_type = typename resource_type::array_type;

  private:
    resource_type m_resource_layout;

  public:
    /** @brief construct from each hierarchy level's extents
      * @param node_dims_ node distribution
      * @param numa_dims_ numa-node distribution within a node
      * @param rank_dims_ rank distribution within a numa-node
      * @param thread_dims_ thread distribution within a rank */
    hierarchical_decomposition(const array_type& node_dims_, const array_type& numa_dims_,
        const array_type& rank_dims_, const array_type& thread_dims_)
    : m_resource_layout(node_dims_, numa_dims_, rank_dims_, thread_dims_)
    {
    }

    hierarchical_decomposition(const hierarchical_decomposition&) = default;
    hierarchical_decomposition& operator=(const hierarchical_decomposition&) = default;

  public:
    /** returns total number of domains */
    size_type size() const noexcept { return m_resource_layout.size(); }

    /** returns number of nodes */
    size_type nodes() const noexcept { return m_resource_layout.template relative_size<0>(); }
    /** returns number of numa nodes per node */
    size_type numas_per_node() const noexcept
    {
        return m_resource_layout.template relative_size<1>();
        ;
    }
    /** returns number of ranks per numa node */
    size_type ranks_per_numa() const noexcept
    {
        return m_resource_layout.template relative_size<2>();
    }
    /** returns number of threads per rank */
    size_type threads_per_rank() const noexcept
    {
        return m_resource_layout.template relative_size<3>();
    }

    /** returns last coordinate in domain */
    array_type last_coord() const noexcept { return m_resource_layout.last_coord(); }

    /** returns node index given a domain index */
    size_type node_index(size_type idx) const noexcept
    {
        return m_resource_layout.template index<0>(idx);
    }
    /** returns numa index given a domain index */
    size_type numa_index(size_type idx) const noexcept
    {
        return m_resource_layout.template index<1>(idx);
    }
    /** returns rank index given a domain index */
    size_type rank_index(size_type idx) const noexcept
    {
        return m_resource_layout.template index<2>(idx);
    }
    /** returns thread index given a domain index */
    size_type thread_index(size_type idx) const noexcept
    {
        return m_resource_layout.template index<3>(idx);
    }

    /** returns node index given rank and thread index */
    size_type node_index(size_type rank, size_type thread_idx) const noexcept
    {
        return node_index(rank * threads_per_rank() + thread_idx);
    }
    /** returns numa index given rank and thread index */
    size_type numa_index(size_type rank, size_type thread_idx) const noexcept
    {
        return numa_index(rank * threads_per_rank() + thread_idx);
    }
    /** returns rank index given rank and thread index */
    size_type rank_index(size_type rank, size_type thread_idx) const noexcept
    {
        return rank_index(rank * threads_per_rank() + thread_idx);
    }

    /** returns node resource index given a domain index */
    size_type node_resource(size_type idx) const noexcept
    {
        return m_resource_layout.template relative_resource<0>(idx);
    }
    /** returns numa resource index given a domain index */
    size_type numa_resource(size_type idx) const noexcept
    {
        return m_resource_layout.template relative_resource<1>(idx);
    }

    /** returns node resource index given rank and thread index */
    size_type node_resource(size_type rank, size_type thread_idx) const noexcept
    {
        return node_resource(rank * threads_per_rank() + thread_idx);
    }
    /** returns numa resource index given rank and thread index */
    size_type numa_resource(size_type rank, size_type thread_idx) const noexcept
    {
        return numa_resource(rank * threads_per_rank() + thread_idx);
    }

    /** returns domain coordinate given a domain index */
    array_type operator()(size_type idx) const noexcept { return m_resource_layout(idx); }

    /** returns domain coordinate given rank and thread index */
    array_type operator()(size_type rank, size_type thread_idx) const noexcept
    {
        return this->operator()(rank* threads_per_rank() + thread_idx);
    }
};

} //namespace ghex
