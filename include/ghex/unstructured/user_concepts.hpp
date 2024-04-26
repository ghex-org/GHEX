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

#include <ghex/config.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/device/cuda/error.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <vector>
#include <utility>
#include <algorithm>
#include <iterator>
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
    using local_index_type = std::size_t;
    using inner_map = std::unordered_map<global_index_type, local_index_type>;
    using outer_map = std::unordered_multimap<global_index_type, local_index_type>;

  private:
    // members
    domain_id_type                 m_id;
    inner_map                      m_inner_map;
    outer_map                      m_outer_map;
    std::vector<global_index_type> m_gids;
    std::vector<global_index_type> m_outer_gids;

  public: // member functions
    domain_id_type                        domain_id() const noexcept { return m_id; }
    std::size_t                           inner_size() const noexcept { return m_inner_map.size(); }
    std::size_t                           size() const noexcept { return m_gids.size(); }
    const inner_map&                      inner_ids() const noexcept { return m_inner_map; }
    const outer_map&                      outer_ids() const noexcept { return m_outer_map; }
    const std::vector<global_index_type>& gids() const noexcept { return m_gids; }
    const std::vector<global_index_type>& outer_gids() const noexcept { return m_outer_gids; }

    std::optional<global_index_type> global_index(local_index_type lid) const noexcept
    {
        return lid < m_gids.size() ? std::optional<global_index_type>{m_gids[lid]} : std::nullopt;
    }

    std::optional<local_index_type> inner_local_index(global_index_type gid) const noexcept
    {
        auto it = m_inner_map.find(gid);
        return it != m_inner_map.end() ? std::optional{it->second} : std::nullopt;
    }

    bool is_inner(global_index_type gid) const noexcept
    {
        return m_inner_map.find(gid) != m_inner_map.end();
    }

    bool is_outer(global_index_type gid) const noexcept
    {
        return m_outer_map.find(gid) != m_outer_map.end();
    }

    // Create a vector of local ids from global ids
    // Repeated gids are allowed iff there are multiple lids mapped to the same gids
    std::vector<local_index_type> make_outer_lids(const std::vector<global_index_type>& gids) const
    {
        std::vector<local_index_type> lids;
        lids.reserve(gids.size());
        std::unordered_map<global_index_type, unsigned int> gid_count;
        for (auto gid : gids)
        {
            auto [first, last] = outer_ids().equal_range(gid);
            if (first == outer_ids().end()) continue;
            if (auto [it, success] = gid_count.insert(std::make_pair(gid, 0u)); !success)
            {
                if (auto c = ++it->second; c < std::distance(first, last)) std::advance(first, c);
                else
                    throw std::runtime_error(
                        "halo gid does not have an associated lid in the domain");
            }
            lids.push_back(first->second);
        }
        for (auto [gid, c] : gid_count)
        {
            auto [first, last] = outer_ids().equal_range(gid);
            if ((c + 1u) != std::distance(first, last))
                throw std::runtime_error("halo gid occurs not often enough");
        }
        return lids;
    }

    // print
    /** @brief print */
    template<typename CharT, typename Traits>
    friend std::basic_ostream<CharT, Traits>& operator<<(std::basic_ostream<CharT, Traits>& os,
        const domain_descriptor&                                                            domain)
    {
        os << "domain id =       " << domain.domain_id() << ";\n"
           << "size =            " << domain.size() << ";\n"
           << "inner size =      " << domain.inner_size() << ";\n"
           << "global ids =       [";
        for (auto x : domain.gids()) os << x << " ";
        os << "]\n"
           << "outer global ids = [";
        for (auto x : domain.outer_gids()) os << x << " ";
        os << "]\n";
        return os;
    }

  public:
    /** @brief Constructs a domain descriptor from a list of global indices and a second list of
      * local indices. The first list describes all elements of the grid, while the latter indicates
      * the position of outer-halo elements within the first list. The order of the global ids is
      * assumed to be the storage order.
      * @tparam GidIt Input iterator to a range of global indices
      * @tparam LidIt Input iterator to a range of local indices
      * @param id Global domain id
      * @param first Iterator pointing to the first global index
      * @param last Iterator pointing to the end (one-past-last) of the global indices
      * @param outer_first Iterator pointing to the first outer local index
      * @param outer_last Iterator pointing to the end (one-past-last) of the outer local indices*/
    template<typename GidIt, typename LidIt>
    domain_descriptor(domain_id_type id, GidIt first, GidIt last, LidIt outer_first,
        LidIt outer_last)
    : m_id{id}
    {
        // temporariliy store the outer local ids in a set for easy retrieval later
        std::unordered_set<local_index_type> outer_lids;
        for (auto it = outer_first; it != outer_last; ++it)
        {
            auto res = outer_lids.insert(*it);
            if (!res.second) throw std::runtime_error("repeated outer (local) index");
        }
        // loop over all global indices and count local index along
        local_index_type lid = 0;
        for (auto it = first; it != last; ++it, ++lid)
        {
            const auto gid = *it;
            if (outer_lids.count(lid))
            {
                m_outer_map.insert(std::make_pair(gid, lid));
                m_outer_gids.push_back(gid);
            }
            else
            {
                auto res = m_inner_map.insert(std::make_pair(gid, lid));
                if (!res.second) throw std::runtime_error("repeated inner (global) index");
            }
            m_gids.push_back(gid);
        }
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
    using local_index_type = typename domain_type::local_index_type;
    using local_indices_type = std::vector<local_index_type>;

    /** @brief Halo concept for unstructured grids
      * TO DO: if everything works, this class definition should be removed,
      * iteration space concept should be moved outside the pattern class,
      * templated on the index type and used here as well.*/
    class halo
    {
      private:
        local_indices_type m_local_indices;

      public:
        // ctors
        halo() noexcept = default;

        halo(local_indices_type local_indices)
        : m_local_indices{std::move(local_indices)}
        {
        }

        // member functions
        /** @brief size of the halo */
        std::size_t               size() const noexcept { return m_local_indices.size(); }
        const local_indices_type& local_indices() const noexcept { return m_local_indices; }

        // print
        /** @brief print */
        template<typename CharT, typename Traits>
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

  private: // members
    std::vector<global_index_type> m_gids;
    bool                           m_use_all = true;

  public: // ctor
    halo_generator() = default;

    halo_generator(std::vector<global_index_type> gids)
    : m_gids{std::move(gids)}
    , m_use_all{false}
    {
    }

    template<typename Iterator>
    halo_generator(Iterator first, Iterator last)
    : m_use_all{false}
    {
        m_gids.insert(m_gids.end(), first, last);
    }

  public: // member functions
    /** @brief generate halo
      * @param domain local domain instance
      * @return receive halo*/
    halo operator()(const domain_type& domain) const
    {
        if (m_use_all) { return {domain.make_outer_lids(domain.outer_gids())}; }
        else { return {domain.make_outer_lids(m_gids)}; }
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
    using byte_t = unsigned char;

  private:
    domain_id_type m_domain_id;
    std::size_t    m_domain_size;
    std::size_t    m_levels;
    value_type*    m_values;
    bool           m_levels_first;
    std::size_t    m_index_stride;
    std::size_t    m_level_stride;

  public:
    // constructors
    // TO DO: check consistency between constructors (const ptr, size and size checks. Here and for the GPU)
    /** @brief constructs a CPU data descriptor using a generic container for the field memory
      * @tparam Container templated container type for the field to be wrapped; data are assumed to
      * be contiguous in memory
      * @param domain local domain instance
      * @param field field to be wrapped
      * @param levels number of levels
      * @param levels_first indicates whether levels have stide 1
      * @param outer_stride outer dimension's stride measured in number of elements of type T (special value 0: no padding)*/
    template<class Container>
    data_descriptor(const domain_descriptor_type& domain, Container& field, std::size_t levels = 1u,
        bool levels_first = true, std::size_t outer_stride = 0u)
    : m_domain_id{domain.domain_id()}
    , m_domain_size{domain.size()}
    , m_levels{levels}
    , m_values{&(field[0])}
    , m_levels_first{levels_first}
    , m_index_stride{levels_first ? (outer_stride ? outer_stride : m_levels) : 1u}
    , m_level_stride{levels_first ? 1u : (outer_stride ? outer_stride : m_domain_size)}
    {
        assert(field.size() == (levels_first ? domain.size() * m_index_stride : m_level_stride * m_levels));
        assert(!(outer_stride) || (outer_stride >= (levels_first ? m_levels : m_domain_size)));
    }

    /** @brief constructs a CPU data descriptor using pointer and size for the field memory
      * @param domain local domain instance
      * @param field_ptr pointer to the field to be wrapped
      * @param levels number of levels
      * @param levels_first stride of levels
      * @param levels_first indicates whether levels have stide 1
      * @param outer_stride outer dimension's stride measured in number of elements of type T (special value 0: no padding)*/
    data_descriptor(const domain_descriptor_type& domain, value_type* field_ptr, std::size_t levels = 1u,
        bool levels_first = true, std::size_t outer_stride = 0u)
    : m_domain_id{domain.domain_id()}
    , m_domain_size{domain.size()}
    , m_levels{levels}
    , m_values{field_ptr}
    , m_levels_first{levels_first}
    , m_index_stride{levels_first ? (outer_stride ? outer_stride : m_levels) : 1u}
    , m_level_stride{levels_first ? 1u : (outer_stride ? outer_stride : m_domain_size)}
    {
        assert(!(outer_stride) || (outer_stride >= (levels_first ? m_levels : m_domain_size)));
    }

    /** @brief constructs a CPU data descriptor using domain parameters and pointer for the field memory
      * @param domain_id local domain id
      * @param domain_size domain size
      * @param field_ptr pointer to the field to be wrapped
      * @param levels number of levels
      * @param levels_first indicates whether levels have stide 1
      * @param outer_stride outer dimension's stride measured in number of elements of type T (special value 0: no padding)*/
    data_descriptor(domain_id_type domain_id, std::size_t domain_size, value_type* field_ptr, std::size_t levels = 1u,
        bool levels_first = true, std::size_t outer_stride = 0u)
    : m_domain_id{domain_id}
    , m_domain_size{domain_size}
    , m_levels{levels}
    , m_values{field_ptr}
    , m_levels_first{levels_first}
    , m_index_stride{levels_first ? (outer_stride ? outer_stride : m_levels) : 1u}
    , m_level_stride{levels_first ? 1u : (outer_stride ? outer_stride : m_domain_size)}
    {
        assert(!(outer_stride) || (outer_stride >= (levels_first ? m_levels : m_domain_size)));
    }

    // member functions

    device_id_type device_id() const noexcept { return arch_traits<arch_type>::default_id(); }

    domain_id_type domain_id() const noexcept { return m_domain_id; }
    std::size_t    domain_size() const noexcept { return m_domain_size; }
    int            num_components() const noexcept { return m_levels; }

    value_type* get_address_at(const std::size_t local_v, const std::size_t level)
    {
        return m_values + (local_v * m_index_stride + level * m_level_stride);
    }

    /** @brief single access operator, used by multiple access set function*/
    value_type& operator()(const std::size_t local_v, const std::size_t level)
    {
        return m_values[local_v * m_index_stride + level * m_level_stride];
    }

    /** @brief single access operator (const version), used by multiple access get function*/
    const value_type& operator()(const std::size_t local_v, const std::size_t level) const
    {
        return m_values[local_v * m_index_stride + level * m_level_stride];
    }

    /** @brief multiple access set function, needed by GHEX to perform the unpacking
      * @tparam IterationSpace iteration space type
      * @param is iteration space which to loop through when setting back the buffer values
      * @param buffer buffer with the data to be set back*/
    template<typename IterationSpace>
    void set(const IterationSpace& is, const byte_t* buffer)
    {
        if (m_levels_first)
        {
            for (std::size_t local_v : is.local_indices())
            {
                for (std::size_t level = 0; level < m_levels; ++level)
                {
                    std::memcpy(&((*this)(local_v, level)), buffer, sizeof(value_type));
                    buffer += sizeof(value_type);
                }
            }
        }
        else
        {
            for (std::size_t level = 0; level < m_levels; ++level)
            {
                for (std::size_t local_v : is.local_indices())
                {
                    std::memcpy(&((*this)(local_v, level)), buffer, sizeof(value_type));
                    buffer += sizeof(value_type);
                }
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
        if (m_levels_first)
        {
            for (std::size_t local_v : is.local_indices())
            {
                for (std::size_t level = 0; level < m_levels; ++level)
                {
                    std::memcpy(buffer, &((*this)(local_v, level)), sizeof(value_type));
                    buffer += sizeof(value_type);
                }
            }
        }
        else
        {
            for (std::size_t level = 0; level < m_levels; ++level)
            {
                for (std::size_t local_v : is.local_indices())
                {
                    std::memcpy(buffer, &((*this)(local_v, level)), sizeof(value_type));
                    buffer += sizeof(value_type);
                }
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
pack_kernel(const T* values, const std::size_t local_indices_size,
    const std::size_t* local_indices, const std::size_t levels, T* buffer,
    const std::size_t index_stride, const std::size_t level_stride,
    const std::size_t buffer_index_stride, const std::size_t buffer_level_stride)
{
    const std::size_t idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (std::size_t level = 0; level < levels; ++level)
        {
            buffer[idx * buffer_index_stride + level * buffer_level_stride] = values[local_indices[idx] * index_stride + level * level_stride];
        }
    }
}

template<typename T>
__global__ void
unpack_kernel(const T* buffer, const std::size_t local_indices_size,
    const std::size_t* local_indices, const std::size_t levels, T* values,
    const std::size_t index_stride, const std::size_t level_stride,
    const std::size_t buffer_index_stride, const std::size_t buffer_level_stride)
{
    const std::size_t idx = threadIdx.x + (blockIdx.x * blockDim.x);
    if (idx < local_indices_size)
    {
        for (std::size_t level = 0; level < levels; ++level)
        {
            values[local_indices[idx] * index_stride + level * level_stride] = buffer[idx * buffer_index_stride + level * buffer_level_stride];
        }
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
    bool           m_levels_first;
    std::size_t    m_index_stride;
    std::size_t    m_level_stride;

  public:
    // constructors
    /** @brief constructs a GPU data descriptor
      * @param domain local domain instance
      * @param field data pointer, assumed to point to a contiguous memory region of size = domain size * n levels
      * @param levels number of levels
      * @param levels_first indicates whether levels have stide 1
      * @param outer_stride outer dimension's stride measured in number of elements of type T (special value 0: no padding)
      * @param device_id device id*/
    data_descriptor(const domain_descriptor_type& domain, value_type* field,
        std::size_t levels = 1u, bool levels_first = true, std::size_t outer_stride = 0u, device_id_type device_id = arch_traits<arch_type>::current_id())
    : m_device_id{device_id}
    , m_domain_id{domain.domain_id()}
    , m_domain_size{domain.size()}
    , m_levels{levels}
    , m_values{field}
    , m_levels_first{levels_first}
    , m_index_stride{levels_first ? (outer_stride ? outer_stride : m_levels) : 1u}
    , m_level_stride{levels_first ? 1u : (outer_stride ? outer_stride : m_domain_size)}
    {
    }

    // member functions

    device_id_type device_id() const noexcept { return m_device_id; }
    domain_id_type domain_id() const noexcept { return m_domain_id; }
    std::size_t    domain_size() const noexcept { return m_domain_size; }
    int            num_components() const noexcept { return m_levels; }

    value_type* get_address_at(const std::size_t local_v, const std::size_t level)
    {
        return m_values + (local_v * m_index_stride + level * m_level_stride);
    }

    template<typename IndexContainer>
    void pack(value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            const int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK));
            const std::size_t buffer_index_stride = m_levels_first ? m_levels : 1u;
            const std::size_t buffer_level_stride = m_levels_first ? 1u : is.local_indices().size();
            pack_kernel<value_type><<<n_blocks, GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK,
                0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(m_values,
                is.local_indices().size(), is.local_indices().data(), m_levels, buffer,
                m_index_stride, m_level_stride, buffer_index_stride, buffer_level_stride);
        }
    }

    template<typename IndexContainer>
    void unpack(const value_type* buffer, const IndexContainer& c, void* stream_ptr)
    {
        for (const auto& is : c)
        {
            const int n_blocks =
                static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) /
                                           GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK));
            const std::size_t buffer_index_stride = m_levels_first ? m_levels : 1u;
            const std::size_t buffer_level_stride = m_levels_first ? 1u : is.local_indices().size();
            unpack_kernel<value_type><<<n_blocks, GHEX_UNSTRUCTURED_SERIALIZATION_THREADS_PER_BLOCK,
                0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(buffer,
                is.local_indices().size(), is.local_indices().data(), m_levels, m_values,
                m_index_stride, m_level_stride, buffer_index_stride, buffer_level_stride);
        }
    }
};

#endif

} // namespace unstructured

} // namespace ghex
