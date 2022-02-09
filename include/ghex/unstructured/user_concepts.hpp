/* 
 * GridTools
 * 
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/device/cuda/error.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iosfwd>

namespace ghex
{
namespace unstructured
{
/** @brief domain descriptor for unstructured domains
 * @tparam DomainId domain id type
 * @tparam Idx global index type*/
template<typename DomainId, typename Idx>
class domain_descriptor
{
  public:
    // member types
    using domain_id_type = DomainId;
    using global_index_type = Idx;
    using vertices_type = std::vector<global_index_type>;
    using vertices_set_type = std::set<global_index_type>;
    using adjncy_type = std::vector<global_index_type>; // named after ParMetis CSR arrays
    using map_type = std::vector<std::pair<global_index_type, adjncy_type>>;
    // local index, deduced from here, and not used at the application level. Will be cast to
    // std::size_t in data_decriptor.
    using local_index_type = std::size_t;

  private:
    // members
    domain_id_type m_id;
    vertices_type  m_vertices; // including halo vertices
    adjncy_type    m_adjncy;   // named after ParMetis CSR arrays
    std::size_t    m_inner_size;
    std::size_t    m_size;
    std::size_t    m_levels;

  public:
    // constructors
    domain_descriptor(const domain_id_type id, const vertices_type& vertices,
        const adjncy_type& adjncy, const std::size_t levels = 1)
    : m_id{id}
    , m_vertices{vertices}
    , m_adjncy{adjncy}
    , m_levels{levels}
    {
        set_halo_vertices();
    }
    domain_descriptor(const domain_id_type id, const map_type& v_map, const std::size_t levels = 1)
    : m_id{id}
    , m_vertices{}
    , m_adjncy{}
    , m_levels{levels}
    {
        for (const auto& v_elem : v_map)
        {
            m_vertices.push_back(v_elem.first);
            m_adjncy.insert(m_adjncy.end(), v_elem.second.begin(), v_elem.second.end());
        }
        set_halo_vertices();
    }
    domain_descriptor(const domain_id_type id, const vertices_type& vertices,
        const std::size_t inner_size, const std::size_t levels = 1)
    : m_id{id}
    , m_vertices{vertices}
    , m_adjncy{} // not set using this constructor. Not a big deal, will eventually be removed
    , m_inner_size{inner_size}
    , m_size{vertices.size()}
    , m_levels{levels}
    {
    }

  private:
    // member functions
    void set_halo_vertices()
    {
        vertices_set_type vertices_set{m_vertices.begin(), m_vertices.end()};
        vertices_set_type all_vertices_set{m_adjncy.begin(), m_adjncy.end()};
        vertices_set_type halo_vertices_set{};
        std::set_difference(all_vertices_set.begin(), all_vertices_set.end(), vertices_set.begin(),
            vertices_set.end(), std::inserter(halo_vertices_set, halo_vertices_set.begin()));
        m_inner_size = m_vertices.size();
        m_vertices.insert(m_vertices.end(), halo_vertices_set.begin(), halo_vertices_set.end());
        m_size = m_vertices.size();
    }

  public:
    // member functions
    domain_id_type       domain_id() const noexcept { return m_id; }
    std::size_t          inner_size() const noexcept { return m_inner_size; }
    std::size_t          size() const noexcept { return m_size; }
    std::size_t          levels() const noexcept { return m_levels; }
    const vertices_type& vertices() const noexcept { return m_vertices; }
    const adjncy_type&   adjncy() const noexcept { return m_adjncy; }

    // print
    /** @brief print */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(
        std::basic_ostream<CharT, Traits>& os, const domain_descriptor& domain)
    {
        os << "domain id = " << domain.domain_id() << ";\n"
           << "inner size = " << domain.inner_size() << ";\n"
           << "size = " << domain.size() << ";\n"
           << "levels = " << domain.levels() << ";\n"
           << "vertices: [ ";
        for (const auto v : domain.vertices()) { os << v << " "; }
        os << "]\n";
        os << "adjncy: [ ";
        for (const auto v : domain.adjncy()) { os << v << " "; }
        os << "]\n";
        return os;
    }
};

/** @brief halo generator for unstructured domains
  * @tparam DomainId domain id type
  * @tparam Idx global index type*/
template<typename DomainId, typename Idx>
class halo_generator
{
  public:
    // member types
    using domain_type = domain_descriptor<DomainId, Idx>;
    using global_index_type = typename domain_type::global_index_type;
    using vertices_type =
        typename domain_type::vertices_type; // mandatory: inferred from the domain
    using local_index_type = typename domain_type::local_index_type;
    using local_indices_type = std::vector<local_index_type>;
    using it_diff_type = typename vertices_type::iterator::difference_type;

    /** @brief Halo concept for unstructured grids
      * TO DO: if everything works, this class definition should be removed,
      * iteration space concept should be moved outside the pattern class,
      * templated on the index type and used here as well.*/
    class halo
    {
      private:
        vertices_type      m_vertices;
        local_indices_type m_local_indices;
        std::size_t        m_levels;

      public:
        // ctors
        halo(const std::size_t levels) noexcept
        : m_levels{levels}
        {
        }
        /** WARN: following one not strictly needed,
          * but it will if this class is used as iteration_space class*/
        halo(const vertices_type& vertices, const local_indices_type& local_indices,
            const std::size_t levels)
        : m_vertices{vertices}
        , m_local_indices{local_indices}
        , m_levels{levels}
        {
        }

        // member functions
        /** @brief size of the halo */
        std::size_t               size() const noexcept { return m_vertices.size(); }
        std::size_t               levels() const noexcept { return m_levels; }
        const vertices_type&      vertices() const noexcept { return m_vertices; }
        const local_indices_type& local_indices() const noexcept { return m_local_indices; }
        void                      push_back(const global_index_type v, const local_index_type idx)
        {
            m_vertices.push_back(v);
            m_local_indices.push_back(idx);
        }

        // print
        /** @brief print */
        template<typename CharT, typename Traits>
        friend std::basic_ostream<CharT, Traits>& operator<<(
            std::basic_ostream<CharT, Traits>& os, const halo& h)
        {
            os << "size = " << h.size() << ";\n"
               << "levels = " << h.levels() << ";\n"
               << "vertices: [ ";
            for (const auto v : h.vertices()) { os << v << " "; }
            os << "]\n"
               << "local indices: [ ";
            for (const auto idx : h.local_indices()) { os << idx << " "; }
            os << "]\n";
            return os;
        }
    };

    // member functions
    /** @brief generate halo
      * @param domain local domain instance
      * @return receive halo*/
    halo operator()(const domain_type& domain) const
    {
        local_indices_type local_indices(domain.size() - domain.inner_size());
        for (size_t i = 0; i < (domain.size() - domain.inner_size()); ++i)
        { local_indices[i] = i + domain.inner_size(); }
        vertices_type vertices{
            domain.vertices().begin() + static_cast<it_diff_type>(domain.inner_size()),
            domain.vertices().end()};
        return {vertices, local_indices, domain.levels()};
    }
};

/** @brief data descriptor for unstructured grids (forward declaration)
  * @tparam Arch device type in which field storage is allocated
  * @tparam DomainId domain id type
  * @tparam Idx global index type
  * @tparam T value type*/
template<typename Arch, typename DomainId, typename Idx, typename T>
class data_descriptor;

/** @brief data descriptor for unstructured grids (CPU specialization)*/
template<typename DomainId, typename Idx, typename T>
class data_descriptor<ghex::cpu, DomainId, Idx, T>
{
  public:
    using arch_type = ghex::cpu;
    using domain_id_type = DomainId;
    using global_index_type = Idx;
    using value_type = T;
    using device_id_type = ghex::arch_traits<arch_type>::device_id_type;
    using domain_descriptor_type = domain_descriptor<domain_id_type, global_index_type>;
    //using allocator_type = std::allocator<value_type>;
    using byte_t = unsigned char;

  private:
    domain_id_type m_domain_id;
    std::size_t    m_domain_size;
    std::size_t    m_levels;
    value_type*    m_values;

  public:
    // constructors
    /** @brief constructs a CPU data descriptor
      * @tparam Container templated container type for the field to be wrapped; data are assumed to
      * be contiguous in memory
      * @param domain local domain instance
      * @param field field to be wrapped*/
    //template<template<typename, typename> class Container>
    template<class Container>
    data_descriptor(
        const domain_descriptor_type& domain, Container& field)
    : m_domain_id{domain.domain_id()}
    , m_domain_size{domain.size()}
    , m_levels{domain.levels()}
    , m_values{&(field[0])}
    {
        assert(field.size() == (domain.size() * domain.levels()));
    }

    // member functions

    device_id_type device_id() const noexcept { return arch_traits<arch_type>::default_id(); } // significant for the GPU
    domain_id_type domain_id() const noexcept { return m_domain_id; }
    std::size_t    domain_size() const noexcept { return m_domain_size; }
    std::size_t    levels() const noexcept { return m_levels; }
    int            num_components() const noexcept { return 1; }

    value_type* get_address_at(const std::size_t local_v, const std::size_t level)
    {
        return m_values + (local_v * m_levels + level);
    }

    /** @brief single access operator, used by multiple access set function*/
    value_type& operator()(const std::size_t local_v, const std::size_t level)
    {
        return m_values[local_v * m_levels + level];
    }

    /** @brief single access operator (const version), used by multiple access get function*/
    const value_type& operator()(const std::size_t local_v, const std::size_t level) const
    {
        return m_values[local_v * m_levels + level];
    }

    /** @brief multiple access set function, needed by GHEX to perform the unpacking
      * @tparam IterationSpace iteration space type
      * @param is iteration space which to loop through when setting back the buffer values
      * @param buffer buffer with the data to be set back*/
    template<typename IterationSpace>
    void set(const IterationSpace& is, const byte_t* buffer)
    {
        for (std::size_t local_v : is.local_indices()) /* TO DO: explicit cast? */
        {
            for (std::size_t level = 0; level < is.levels(); ++level)
            {
                std::memcpy(&((*this)(local_v, level)), buffer, sizeof(value_type));
                buffer += sizeof(value_type);
            }
        }
    }

    /** @brief multiple access get function, needed by GHEX to perform the packing
      * @tparam IterationSpace iteration space type
      * @param is iteration space which to loop through when getting the data from the internal storage
      * @param buffer buffer to be filled*/
    template<typename IterationSpace>
    void get(const IterationSpace& is, byte_t* buffer) const
    {
        for (std::size_t local_v : is.local_indices()) /* TO DO: explicit cast? */
        {
            for (std::size_t level = 0; level < is.levels(); ++level)
            {
                std::memcpy(buffer, &((*this)(local_v, level)), sizeof(value_type));
                buffer += sizeof(value_type);
            }
        }
    }

    template<typename IndexContainer>
    void pack(value_type* buffer, const IndexContainer& c, void*)
    {
        for (const auto& is : c) { get(is, reinterpret_cast<byte_t*>(buffer)); }
    }

    template<typename IndexContainer>
    void unpack(const value_type* buffer, const IndexContainer& c, void*)
    {
        for (const auto& is : c) { set(is, reinterpret_cast<const byte_t*>(buffer)); }
    }
};

#ifdef GHEX_CUDACC

#define GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK 32

template<typename T>
__global__ void
pack_kernel(const T* values, const std::size_t local_indices_size, const std::size_t* local_indices,
    const std::size_t levels, T* buffer)
{
    std::size_t idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (std::size_t level = 0; level < levels; ++level)
        { buffer[idx * levels + level] = values[local_indices[idx] * levels + level]; }
    }
}

template<typename T>
__global__ void
unpack_kernel(const T* buffer, const std::size_t local_indices_size,
    const std::size_t* local_indices, const std::size_t levels, T* values)
{
    std::size_t idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (std::size_t level = 0; level < levels; ++level)
        { values[local_indices[idx] * levels + level] = buffer[idx * levels + level]; }
    }
}

/** @brief data descriptor for unstructured grids (GPU specialization)*/
template<typename DomainId, typename Idx, typename T>
class data_descriptor<gpu, DomainId, Idx, T>
{
  public:
    using arch_type = gpu;
    using domain_id_type = DomainId;
    using global_index_type = Idx;
    using value_type = T;
    using device_id_type = arch_traits<arch_type>::device_id_type;
    using domain_descriptor_type = domain_descriptor<domain_id_type, global_index_type>;

  private:
    device_id_type m_device_id;
    domain_id_type m_domain_id;
    std::size_t    m_domain_size;
    std::size_t    m_levels;
    value_type*    m_values;

  public:
    // constructors
    /** @brief constructs a GPU data descriptor
      * @param domain local domain instance
      * @param field data pointer, assumed to point to a contiguous memory region of size = domain size * n levels
      * @param device_id device id*/
    data_descriptor(
        const domain_descriptor_type& domain, value_type* field, device_id_type device_id = arch_traits<arch_type>::current_id())
    : m_device_id{device_id}
    , m_domain_id{domain.domain_id()}
    , m_domain_size{domain.size()}
    , m_levels{domain.levels()}
    , m_values{field}
    {
    }

    // member functions

    device_id_type device_id() const noexcept { return m_device_id; }
    domain_id_type domain_id() const noexcept { return m_domain_id; }
    std::size_t    domain_size() const noexcept { return m_domain_size; }
    std::size_t    levels() const noexcept { return m_levels; }
    int            num_components() const noexcept { return 1; }

    value_type* get_address_at(const std::size_t local_v, const std::size_t level)
    {
        return m_values + (local_v * m_levels + level);
    }

    /** @brief single access operator
                 * used from the host to get data pointers, e.g.: &(data(i, j))*/ // TO DO: find better solution
    value_type& operator()(const std::size_t local_v, const std::size_t level)
    {
        return m_values[local_v * m_levels + level];
    }

    /** @brief single access operator (const version)
                 * used from the host to get data pointers, e.g.: &(data(i, j))*/ // TO DO: find better solution
    const value_type& operator()(const std::size_t local_v, const std::size_t level) const
    {
        return m_values[local_v * m_levels + level];
    }

    template<typename IndexContainer>
    void pack(value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK));
            pack_kernel<value_type><<<n_blocks, GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK,
                0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                m_values, is.local_indices().size(), &(is.local_indices()[0]), is.levels(), buffer);
        }
    }

    template<typename IndexContainer>
    void unpack(const value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK));
            unpack_kernel<value_type><<<n_blocks, GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK,
                0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                buffer, is.local_indices().size(), &(is.local_indices()[0]), is.levels(), m_values);
        }
    }
};

#endif

} // namespace unstructured

} // namespace ghex
