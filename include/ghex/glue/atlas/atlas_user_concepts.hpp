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

#include <atlas/field.h>
#include <atlas/array.h>

#include <ghex/config.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/device/cuda/error.hpp>
#include <ghex/glue/atlas/field.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iosfwd>
#include <utility>

namespace ghex
{
/** @brief Implements domain descriptor concept for Atlas domains
  * An Atlas domain is assumed to include the halo region as well,
  * and has therefore to be istantiated using a mesh which has already grown the required halo layer
  * after the creation of a function space with a halo.
  * Null halo is fine too, provided that the mesh is in its final state.
  * Domain size includes halo size.
  * @tparam DomainId domain id type*/
template<typename DomainId>
class atlas_domain_descriptor
{
  public:
    // member types
    using domain_id_type = DomainId;
    using local_index_type = ::atlas::idx_t;

  private:
    // members
    domain_id_type   m_id;
    ::atlas::Field   m_partition;
    ::atlas::Field   m_remote_index;
    local_index_type m_size;

  public:
    // ctors
    /** @brief Constructs a local domain
      * @param id domain id
      * @param partition partition indices of domain (+ halo) elements (Atlas field)
      * @param remote_index local indices in remote partition for domain (+ halo) elements (Atlas field)*/
    atlas_domain_descriptor(const domain_id_type id, const ::atlas::Field& partition,
        const ::atlas::Field& remote_index)
    : m_id{id}
    , m_partition{partition}
    , m_remote_index{remote_index}
    , m_size{static_cast<local_index_type>(partition.size())}
    {
        // Asserts
        assert(partition.size() == remote_index.size());
    }

    // member functions
    domain_id_type        domain_id() const noexcept { return m_id; }
    const ::atlas::Field& partition() const noexcept { return m_partition; }
    const ::atlas::Field& remote_index() const noexcept { return m_remote_index; }
    local_index_type      size() const noexcept { return m_size; }

    // print
    /** @brief print */
    template<class CharT, class Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
        const atlas_domain_descriptor&                                                      domain)
    {
        os << "domain id = " << domain.domain_id() << ";\n"
           << "size = " << domain.size() << ";\n"
           << "partition indices: [" << domain.partition() << "]\n"
           << "remote indices: [" << domain.remote_index() << "]\n";
        return os;
    }
};

/** @brief halo generator for atlas domains
  * An Atlas domain has already the notion of halos.
  * The halo generator isolates the indices referring to the halo points.
  * @tparam DomainId domain id type*/
template<typename DomainId>
class atlas_halo_generator
{
  public:
    // member types
    using domain_type = atlas_domain_descriptor<DomainId>;
    using local_index_type = typename domain_type::local_index_type;

    /** @brief Halo class for Atlas
     * Provides list of local indices of neighboring elements.*/
    class halo
    {
      private:
        std::vector<local_index_type> m_local_indices;

      public:
        // ctors
        halo() noexcept = default;

        // member functions
        std::size_t                    size() const noexcept { return m_local_indices.size(); }
        std::vector<local_index_type>& local_indices() noexcept { return m_local_indices; }
        const std::vector<local_index_type>& local_indices() const noexcept
        {
            return m_local_indices;
        }

        // print
        /** @brief print */
        template<class CharT, class Traits>
        friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
            const halo&                                                                         h)
        {
            os << "size = " << h.size() << ";\n"
               << "local indices: [ ";
            for (const auto idx : h.local_indices()) { os << idx << " "; }
            os << "]\n";
            return os;
        }
    };

  public:
    // member functions
    /** @brief generates the halo
     * @param domain local domain instance
     * @return receive halo*/
    halo operator()(const domain_type& domain) const
    {
        auto partition = ::atlas::array::make_view<int, 1>(domain.partition());
        auto remote_index = ::atlas::array::make_view<local_index_type, 1>(domain.remote_index());

        halo h;

        // if the index refers to another domain, or even to the same but as a halo point,
        // the halo is updated
        for (local_index_type d_idx = 0; d_idx < domain.size(); ++d_idx)
        {
            if ((partition(d_idx) != domain.domain_id()) || (remote_index(d_idx) != d_idx))
            {
                h.local_indices().push_back(d_idx);
            }
        }

        return h;
    }
};

/** @brief recv domain ids generator for atlas domains
  * The recv domain ids generator isolates the domain ids referring to the halo points,
  * together with their remote indices and the ranks they belong to.
  * Atlas will always assume no oversubscription, and domain id == rank id.
  * @tparam DomainId domain id type*/
template<typename DomainId>
class atlas_recv_domain_ids_gen
{
  public:
    // member types
    using domain_id_type = DomainId;
    using domain_type = atlas_domain_descriptor<domain_id_type>;
    using local_index_type = typename domain_type::local_index_type;

    /** @brief Halo class for Atlas recv domain ids generator
      * Provides following lists, each of which corresponds to the list of halo points:
      * - receive domain ids;
      * - indices of halo points on remote domains (remote indices);
      * - ranks which the domains belongs to (no oversubscription). */
    class halo
    {
      private:
        std::vector<domain_id_type>   m_domain_ids;
        std::vector<local_index_type> m_remote_indices;
        std::vector<int>              m_ranks;

      public:
        // member functions
        std::vector<domain_id_type>&         domain_ids() noexcept { return m_domain_ids; }
        const std::vector<domain_id_type>&   domain_ids() const noexcept { return m_domain_ids; }
        std::vector<local_index_type>&       remote_indices() noexcept { return m_remote_indices; }
        const std::vector<local_index_type>& remote_indices() const noexcept
        {
            return m_remote_indices;
        }
        std::vector<int>&       ranks() noexcept { return m_ranks; }
        const std::vector<int>& ranks() const noexcept { return m_ranks; }

        // print
        /** @brief print */
        template<class CharT, class Traits>
        friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
            const halo&                                                                         h)
        {
            os << "domain ids: [ ";
            for (auto d_id : h.domain_ids()) { os << d_id << " "; }
            os << "]\n";
            os << "remote indices: [";
            for (auto r_idx : h.remote_indices()) { os << r_idx << " "; }
            os << "]\n";
            os << "ranks: [";
            for (auto r : h.ranks()) { os << r << " "; }
            os << "]\n";
            return os;
        }
    };

  public:
    // member functions
    /** @brief generates halo with receive domain ids
      * @param domain local domain instance
      * @return receive domain ids halo*/
    halo operator()(const domain_type& domain) const
    {
        auto partition = ::atlas::array::make_view<int, 1>(domain.partition());
        auto remote_index = ::atlas::array::make_view<local_index_type, 1>(domain.remote_index());

        halo h{};

        // if the index refers to another domain, or even to the same but as a halo point,
        // the halo is updated
        for (local_index_type d_idx = 0; d_idx < domain.size(); ++d_idx)
        {
            if ((partition(d_idx) != domain.domain_id()) || (remote_index(d_idx) != d_idx))
            {
                h.domain_ids().push_back(partition(d_idx));
                h.remote_indices().push_back(remote_index(d_idx));
                h.ranks().push_back(static_cast<int>(partition(d_idx)));
            }
        }

        return h;
    }
};

/** @brief Atlas data descriptor (forward declaration)
  * @tparam Arch device type in which field storage is allocated
  * @tparam DomainId domain id type
  * @tparam T value type*/
template<typename Arch, typename DomainId, typename T, typename StorageTraits,
    typename FunctionSpace>
class atlas_data_descriptor;

/** @brief Atlas data descriptor (CPU specialization)*/
template<typename DomainId, typename T, typename StorageTraits, typename FunctionSpace>
class atlas_data_descriptor<ghex::cpu, DomainId, T, StorageTraits, FunctionSpace>
{
  public:
    using arch_type = ghex::cpu;
    using domain_id_type = DomainId;
    using value_type = T;
    using storage_traits_type = StorageTraits;
    using function_space_type = FunctionSpace;
    using domain_descriptor_type = atlas_domain_descriptor<domain_id_type>;
    using local_index_type = typename domain_descriptor_type::local_index_type;
    using device_id_type = ghex::arch_traits<arch_type>::device_id_type;
    using byte_t = unsigned char;
    using field_type = atlas::field<value_type, storage_traits_type, function_space_type>;
    using view_type = decltype(std::declval<field_type>().target_view());

  private:
    domain_id_type m_domain_id;
    view_type&     m_values;
    int m_components; // TO DO: idx_t? Fix also usage in operator(), set() and get() below

  public:
    /** @brief constructs a CPU data descriptor
      * @param domain local domain instance
      * @param view field view to be wrapped
      * @param components number of field components */
    atlas_data_descriptor(const domain_descriptor_type& domain, view_type& values,
        const int components)
    : m_domain_id{domain.domain_id()}
    , m_values{values}
    , m_components{components}
    {
    }

    domain_id_type domain_id() const { return m_domain_id; }

    device_id_type device_id() const { return arch_traits<arch_type>::default_id(); }

    int num_components() const noexcept { return m_components; }

    /** @brief single access operator, used by multiple access set function*/
    value_type& operator()(const local_index_type idx, const int component)
    {
        return m_values(idx, component);
    }

    /** @brief single access operator (const version), used by multiple access get function*/
    const value_type& operator()(const local_index_type idx, const int component) const
    {
        return m_values(idx, component);
    }

    /** @brief multiple access set function, needed by GHEX to perform the unpacking
      * @tparam IterationSpace iteration space type
      * @param is iteration space which to loop through when setting back the buffer values
      * @param buffer buffer with the data to be set back*/
    template<typename IterationSpace>
    void set(const IterationSpace& is, const byte_t* buffer)
    {
        for (local_index_type idx : is.local_indices())
        {
            for (int component = 0; component < m_components; ++component)
            {
                std::memcpy(&((*this)(idx, component)), buffer, sizeof(value_type));
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
        for (local_index_type idx : is.local_indices())
        {
            for (int component = 0; component < m_components; ++component)
            {
                std::memcpy(buffer, &((*this)(idx, component)), sizeof(value_type));
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

#define GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK 32

template<typename T, typename View, typename Index> // TO DO: in principle, Field would be enough
__global__ void
pack_kernel(const View values, const std::size_t local_indices_size, const Index* local_indices,
    const int components, T* buffer)
{
    auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (int component = 0; component < components; ++component)
        {
            buffer[idx * components + component] = values(local_indices[idx], component);
        }
    }
}

template<typename T, typename View, typename Index> // TO DO: in principle, Field would be enough
__global__ void
unpack_kernel(const T* buffer, const std::size_t local_indices_size, const Index* local_indices,
    const int components, View values)
{
    auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (int component = 0; component < components; ++component)
        {
            values(local_indices[idx], component) = buffer[idx * components + component];
        }
    }
}

/** @brief Atlas data descriptor (GPU specialization)*/
template<typename DomainId, typename T, typename StorageTraits, typename FunctionSpace>
class atlas_data_descriptor<ghex::gpu, DomainId, T, StorageTraits, FunctionSpace>
{
  public:
    using arch_type = ghex::gpu;
    using domain_id_type = DomainId;
    using value_type = T;
    using storage_traits_type = StorageTraits;
    using function_space_type = FunctionSpace;
    using domain_descriptor_type = atlas_domain_descriptor<domain_id_type>;
    using local_index_type = typename domain_descriptor_type::local_index_type;
    using device_id_type = ghex::arch_traits<arch_type>::device_id_type;
    using field_type = atlas::field<value_type, storage_traits_type, function_space_type>;
    using view_type = decltype(std::declval<field_type>().target_view());

  private:
    domain_id_type m_domain_id;
    device_id_type m_device_id;
    view_type&     m_values;
    int m_components; // TO DO: idx_t? Fix also usage in operator(), set() and get() below

  public:
    /** @brief constructs a GPU data descriptor
      * @param domain local domain instance
      * @param device_id device id
      * @param field field to be wrapped*/
    atlas_data_descriptor(const domain_descriptor_type& domain, const device_id_type device_id,
        view_type& values, const int components)
    : m_domain_id{domain.domain_id()}
    , m_device_id{device_id}
    , m_values{values}
    , m_components{components}
    {
    }

    /** @brief data type size, mandatory*/
    std::size_t data_type_size() const { return sizeof(value_type); }

    domain_id_type domain_id() const noexcept { return m_domain_id; }

    device_id_type device_id() const noexcept { return m_device_id; }

    int num_components() const noexcept { return m_components; }

    template<typename IndexContainer>
    void pack(value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
            pack_kernel<<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0,
                *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(m_values,
                is.local_indices().size(), is.local_indices().data(), m_components, buffer);
        }
    }

    template<typename IndexContainer>
    void unpack(const value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
            unpack_kernel<<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0,
                *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(buffer, is.local_indices().size(),
                is.local_indices().data(), m_components, m_values);
        }
    }
};

#endif

} // namespace ghex
