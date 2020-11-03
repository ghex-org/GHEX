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
#ifndef INCLUDED_GHEX_STRUCTURED_REGULAR_DECOMPOSITION_HPP
#define INCLUDED_GHEX_STRUCTURED_REGULAR_DECOMPOSITION_HPP

#include "../../distribution.hpp"

namespace gridtools {
namespace ghex {
namespace structured {
namespace regular {

// A class which stores a hierarchical domain decomposition.
// The hierarchy is established through 4 levels which can be mapped to
// - nodes per run
// - numa-nodes per node
// - ranks per numa-node
// - threads per rank
//
// Each level stores the extents of a D-dimensional hypercube. Given a scalar index
// or a rank-index, thread-index pair it computes the coordinate in D-dimensional space of the
// corresponding domain. The relative order within each level is Fortran / first dimension has
// stride=1.
template<unsigned int D>
class hierarchical_decomposition
{
private:
    using distribution_type = hierarchical_distribution<4>;
    using dims_map_type = dims_map<D>;

public:
    using size_type = typename dims_map_type::size_type;
    using array_type = typename dims_map_type::array_type;

private:
    dims_map_type m_node_dims;
    dims_map_type m_numa_dims;
    dims_map_type m_rank_dims;
    dims_map_type m_thread_dims;
    distribution_type m_dist;
    hierarchical_distribution<3> m_node_dist;
    hierarchical_distribution<2> m_numa_dist;

public:
    hierarchical_decomposition(
        const array_type& node_dims_,
        const array_type& numa_dims_,
        const array_type& rank_dims_,
        const array_type& thread_dims_)
    : m_node_dims(node_dims_,false)
    , m_numa_dims(numa_dims_,false)
    , m_rank_dims(rank_dims_,false)
    , m_thread_dims(thread_dims_,false)
    , m_dist({m_node_dims.size(), m_numa_dims.size(), m_rank_dims.size(), m_thread_dims.size()},true)
    , m_node_dist({m_numa_dims.size(), m_rank_dims.size(), m_thread_dims.size()},true)
    , m_numa_dist({m_rank_dims.size(), m_thread_dims.size()},true)
    {}

    hierarchical_decomposition(const hierarchical_decomposition&) = default;
    hierarchical_decomposition& operator=(const hierarchical_decomposition&) = default;

public:
    // returns number of threads per rank
    size_type threads_per_rank() const noexcept { return m_thread_dims.size(); }
    // returns number of ranks per numa node
    size_type ranks_per_numa() const noexcept { return m_rank_dims.size(); }
    // returns number of numa nodes per node
    size_type numas_per_node() const noexcept { return m_numa_dims.size(); }
    // returns number of nodes
    size_type nodes() const noexcept { return m_node_dims.size(); }
    // returns total number of domains
    size_type size() const noexcept { return m_dist.size(); }
    // returns last coordinate in domain
    array_type last_coord() const noexcept { return this->operator()(size()-1); }
    // returns node index given a domain index
    size_type node_index(size_type idx) const noexcept { return m_dist(idx)[0]; }
    // returns numa index given a domain index
    size_type numa_index(size_type idx) const noexcept { return m_dist(idx)[1]; }
    // returns rank index given a domain index
    size_type rank_index(size_type idx) const noexcept { return m_dist(idx)[2]; }
    // returns node index given rank and thread index
    size_type node_index(size_type rank, size_type thread_idx) const noexcept
    {
        return node_index(rank*threads_per_rank()+thread_idx);
    }
    // returns numa index given rank and thread index
    size_type numa_index(size_type rank, size_type thread_idx) const noexcept
    {
        return numa_index(rank*threads_per_rank()+thread_idx);
    }
    // returns rank index given rank and thread index
    size_type rank_index(size_type rank, size_type thread_idx) const noexcept
    {
        return rank_index(rank*threads_per_rank()+thread_idx);
    }

    // returns node resource index given a domain index
    size_type node_resource(size_type idx) const noexcept
    {
        return idx - node_index(idx)*m_node_dist.size();
    }
    // returns numa resource index given a domain index
    size_type numa_resource(size_type idx) const noexcept
    {
        return node_resource(idx) - numa_index(idx)*m_numa_dist.size();
    }
    // returns node resource index given rank and thread index
    size_type node_resource(size_type rank, size_type thread_idx) const noexcept
    {
        return node_resource(rank*threads_per_rank()+thread_idx);
    }
    // returns numa resource index given rank and thread index
    size_type numa_resource(size_type rank, size_type thread_idx) const noexcept
    {
        return numa_resource(rank*threads_per_rank()+thread_idx);
    }

    // returns domain coordinate given a domain index
    array_type operator()(size_type idx) const noexcept
    {
        auto indices = m_dist(idx);
        auto node_coord = m_node_dims(indices[0]);
        auto numa_coord = m_numa_dims(indices[1]);
        auto rank_coord = m_rank_dims(indices[2]);
        auto thread_coord = m_thread_dims(indices[3]);
        array_type res;
        for (unsigned i=0; i<D; ++i)
            res[i] = 
                thread_coord[i]
                    + m_thread_dims.dims()[i]*(rank_coord[i]
                        + m_rank_dims.dims()[i]*(numa_coord[i]
                            + node_coord[i]*m_numa_dims.dims()[i]));
        return res;
    }
    
    // returns domain coordinate given rank and thread index
    array_type operator()(size_type rank, size_type thread_idx) const noexcept
    {
        return this->operator()(rank*threads_per_rank()+thread_idx);
    }
};

} //namespace regular
} //namespace structured
} //namespace ghex
} //namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_REGULAR_DECOMPOSITION_HPP */
