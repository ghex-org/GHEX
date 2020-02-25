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
#ifndef INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP
#define INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP

#include <vector>
#include <cassert>
#include <cstring>
#include <cmath>
#include <iosfwd>

#include <atlas/field.h>
#include <atlas/array.h>

#include "../../unstructured/grid.hpp"
#include "../../arch_list.hpp"
#include "../../arch_traits.hpp"
#include "../../allocator/unified_memory_allocator.hpp"

#ifdef __CUDACC__
#include "../../cuda_utils/error.hpp"
#endif


namespace gridtools {

    namespace ghex {

        /** @brief Implements domain descriptor concept for Atlas domains
         * An Atlas domain is assumed to inlude the halo region as well,
         * and has therefore to be istantiated using a mesh which has already grown the required halo layer
         * after the creation of a function space with a halo.
         * Null halo is fine too, provided that the mesh is in its final state.
         * Domain size is given including halo size.
         * @tparam DomainId domain id type*/
        template<typename DomainId>
        class atlas_domain_descriptor {

            public:

                // member types
                using domain_id_type = DomainId;
                using index_t = atlas::idx_t;

            private:

                // members
                domain_id_type m_id;
                int m_rank;
                atlas::Field m_partition;
                atlas::Field m_remote_index;
                std::size_t m_levels;
                index_t m_size;
                index_t m_first;
                index_t m_last;

            public:

                // ctors
                /** @brief Constructs a local domain
                 * @param id domain id
                 * @param rank rank of processing element
                 * @param partition partition indices of domain (+ halo) elements (Atlas field)
                 * @param remote_index local indices in remote partition for domain (+ halo) elements (Atlas field)
                 * @param levels number of vertical levels
                 * @param size number of domain + halo points*/
                atlas_domain_descriptor(domain_id_type id,
                                        const int rank,
                                        const atlas::Field& partition,
                                        const atlas::Field& remote_index,
                                        const std::size_t levels,
                                        const index_t size) :
                    m_id{id},
                    m_rank{rank},
                    m_partition{partition},
                    m_remote_index{remote_index},
                    m_levels{levels},
                    m_size{size} {

                    // Asserts
                    assert(size > 0);
                    assert(size == partition.size());
                    assert(size == remote_index.size());

                    // Setup first and last (with the assumption that first/last coincides with min/max)
                    const int* partition_data = atlas::array::make_view<int, 1>(m_partition).data();
                    const index_t* remote_index_data = atlas::array::make_view<index_t, 1>(m_remote_index).data();
                    index_t first = remote_index_data[0];
                    index_t last = remote_index_data[0];
                    for (auto idx = 1; idx < m_size; ++idx) {
                        if (partition_data[idx] == rank) {
                            if (remote_index_data[idx] < first) {
                                first = remote_index_data[idx];
                            } else if (remote_index_data[idx] > last) {
                                last = remote_index_data[idx];
                            }
                        }
                    }
                    m_first = first;
                    m_last = last;

                }

                // member functions
                domain_id_type domain_id() const noexcept { return m_id; }
                int rank() const noexcept { return m_rank; }
                const atlas::Field& partition() const noexcept { return m_partition; }
                const atlas::Field& remote_index() const noexcept { return m_remote_index; }
                std::size_t levels() const noexcept { return m_levels; }
                index_t size() const noexcept { return m_size; }
                /** @brief first local index, excluding halo points*/
                index_t first() const noexcept { return m_first; }
                /** @brief last local index, excluding halo points*/
                index_t last() const noexcept { return m_last; }

                // print
                /** @brief print */
                template<class CharT, class Traits>
                friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const atlas_domain_descriptor& domain) {
                    os << "domain id = " << domain.domain_id() << ";\n"
                       << "size = " << domain.size() << ";\n"
                       << "# levels = " << domain.levels() << ";\n"
                       << "partition indices: [" << domain.partition() << "]\n"
                       << "remote indices: [" << domain.remote_index() << "]\n";
                    return os;
                }

        };

        /** @brief halo generator for atlas domains
         * An Atlas domain has already the notion of halos.
         * The purpose of the halo generator is to fill a container
         * in which each element (halo) gathers all the indices
         * referring to the same remote partition.
         * @tparam DomainId domain id type*/
        template<typename DomainId>
        class atlas_halo_generator {

            public:

                // member types
                using domain_type = atlas_domain_descriptor<DomainId>;
                using index_t = atlas::idx_t;
                using u_m_allocator_t = gridtools::ghex::allocator::cuda::unified_memory_allocator<index_t>;

                /** @brief Halo concept for Atlas
                 * Puts together a neighbouring partition index
                 * and the sequences of indices of the elements of that partition.
                 * Conceptually it is similar to an iteration space
                 * and the two concepts could potentially coincide.
                 * Stores information on the number of vertical levels as well,
                 * but its size corresponds only to the horizontal size.*/
                class halo {

                    private:

                        int m_partition;
                        std::vector<index_t, u_m_allocator_t> m_local_index;
                        std::vector<index_t> m_remote_index;
                        std::size_t m_levels;

                    public:

                        // ctors
                        halo() noexcept = default;
                        halo(const int partition, const std::size_t levels = 1) noexcept :
                            m_partition{partition},
                            m_local_index{},
                            m_remote_index{},
                            m_levels{levels} {}
                        // following one not strictly needed
                        halo(const int partition,
                             const std::vector<index_t, u_m_allocator_t>& local_index,
                             const std::vector<index_t>& remote_index,
                             const std::size_t levels = 1) noexcept :
                            m_partition{partition},
                            m_local_index{local_index},
                            m_remote_index{remote_index},
                            m_levels{levels} {
                            assert(local_index.size() == remote_index.size());
                        }

                        // member functions
                        int partition() const noexcept { return m_partition; }
                        std::vector<index_t, u_m_allocator_t>& local_index() noexcept { return m_local_index; }
                        const std::vector<index_t, u_m_allocator_t>& local_index() const noexcept { return m_local_index; }
                        std::vector<index_t>& remote_index() noexcept { return m_remote_index; }
                        const std::vector<index_t>& remote_index() const noexcept { return m_remote_index; }
                        std::size_t levels() const noexcept { return m_levels; }
                        std::size_t size() const noexcept { return m_local_index.size(); }
                        void push_back(const index_t local_index_idx, const index_t remote_index_idx) {
                            m_local_index.push_back(local_index_idx);
                            m_remote_index.push_back(remote_index_idx);
                        }

                        // print
                        /** @brief print */
                        template<class CharT, class Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const halo& h) {
                            os << "size = " << h.size() << ";\n"
                               << "# levels = " << h.levels() << ";\n"
                               << "partition = " << h.partition() << ";\n"
                               << "local indices: [ ";
                            for (auto idx : h.local_index()) { os << idx << " "; }
                            os << "]\nremote indices: [ ";
                            for (auto idx : h.remote_index()) { os << idx << " "; }
                            os << "]\n";
                            return os;
                        }

                };

            private:

                //members
                int m_size;

            public:

                // ctors
                /** @brief constructs a halo generator
                 * @param size number of processing elements*/
                atlas_halo_generator(const int size) noexcept : m_size{size} {}

                // member functions
                /** @brief generate halos (assumes 1 local domain per processing element)
                 * @param domain local domain instance
                 * @return vector of receive halos, one for each local domain*/
                auto operator()(const domain_type& domain) const {

                    const int* partition_data = atlas::array::make_view<int, 1>(domain.partition()).data();
                    const index_t* remote_index_data = atlas::array::make_view<index_t, 1>(domain.remote_index()).data();

                    std::vector<halo> halos{};
                    for (int rank = 0; rank < m_size; ++rank) {
                        halos.push_back({rank, domain.levels()});
                    }

                    // if the index refers to another rank, or even to the same rank but as a halo point,
                    // the corresponding halo is updated
                    for (auto domain_idx = 0; domain_idx < domain.size(); ++domain_idx) {
                        if ((partition_data[domain_idx] != domain.rank()) || (remote_index_data[domain_idx] != domain_idx)) {
                            halos[partition_data[domain_idx]].push_back(domain_idx, remote_index_data[domain_idx]);
                        }
                    }

                    return halos;

                }

        };

        /** @brief Atlas data descriptor (forward declaration)
         * @tparam Arch device type in which field storage is allocated
         * @tparam DomainId domain id type
         * @tparam T value type*/
        template <typename Arch, typename DomainId, typename T>
        class atlas_data_descriptor;

        /** @brief Atlas data descriptor (CPU specialization)*/
        template <typename DomainId, typename T>
        class atlas_data_descriptor<gridtools::ghex::cpu, DomainId, T> {

            public:

                using arch_type = gridtools::ghex::cpu;
                using domain_id_type = DomainId;
                using value_type = T;
                using domain_descriptor_type = atlas_domain_descriptor<domain_id_type>;
                using index_t = typename domain_descriptor_type::index_t;
                using device_id_type = gridtools::ghex::arch_traits<arch_type>::device_id_type;
                using byte_t = unsigned char;

            private:

                domain_id_type m_domain_id;
                atlas::array::ArrayView<value_type, 2> m_values;

            public:

                /** @brief constructs a CPU data descriptor
                 * @param domain local domain instance
                 * @param field Atlas field to be wrapped*/
                atlas_data_descriptor(const domain_descriptor_type& domain,
                                      const atlas::Field& field) :
                    m_domain_id{domain.domain_id()},
                    m_values{atlas::array::make_view<value_type, 2>(field)} {}

                /** @brief data type size, mandatory*/
                std::size_t data_type_size() const {
                    return sizeof (value_type);
                }

                domain_id_type domain_id() const { return m_domain_id; }

                device_id_type device_id() const { return 0; }

                /** @brief single access operator, used by multiple access set function*/
                value_type& operator()(const index_t idx, const std::size_t level) {
                    return m_values(static_cast<std::size_t>(idx), level); // WARN: why std::size_t cast is needed here?
                }

                /** @brief single access operator (const version), used by multiple access get function*/
                const value_type& operator()(const index_t idx, const std::size_t level) const {
                    return m_values(static_cast<std::size_t>(idx), level); // WARN: why std::size_t cast is needed here?
                }

                /** @brief multiple access set function, needed by GHEX to perform the unpacking
                 * @tparam IterationSpace iteration space type
                 * @param is iteration space which to loop through when setting back the buffer values
                 * @param buffer buffer with the data to be set back*/
                template <typename IterationSpace>
                void set(const IterationSpace& is, const byte_t* buffer) {
                    for (index_t idx : is.local_index()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            std::memcpy(&((*this)(idx, level)), buffer, sizeof(value_type));
                            buffer += sizeof(value_type);
                        }
                    }
                }

                /** @brief multiple access get function, needed by GHEX to perform the packing
                 * @tparam IterationSpace iteration space type
                 * @param is iteration space which to loop through when getting the data from the internal storage
                 * @param buffer buffer to be filled*/
                template <typename IterationSpace>
                void get(const IterationSpace& is, byte_t* buffer) const {
                    for (index_t idx : is.local_index()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            std::memcpy(buffer, &((*this)(idx, level)), sizeof(value_type));
                            buffer += sizeof(value_type);
                        }
                    }
                }

                template<typename IndexContainer>
                void pack(value_type* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        get(is, reinterpret_cast<byte_t*>(buffer));
                    }
                }

                template<typename IndexContainer>
                void unpack(const value_type* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        set(is, reinterpret_cast<const byte_t*>(buffer));
                    }
                }

        };

#ifdef __CUDACC__

#define GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK 32

        template <typename T, typename index_t>
        __global__ void pack_kernel(
                const atlas::array::ArrayView<T, 2> values,
                const std::size_t local_index_size,
                const index_t* local_index,
                const std::size_t levels,
                const std::size_t buffer_size,
                T* buffer) {
            auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < local_index_size) {
                for(auto level = 0; level < levels; ++level) {
                    buffer[idx * levels + level] = values(local_index[idx], level);
                }
            }
        }

        template <typename T, typename index_t>
        __global__ void unpack_kernel(
                const std::size_t buffer_size,
                const T* buffer,
                const std::size_t local_index_size,
                const index_t* local_index,
                const std::size_t levels,
                atlas::array::ArrayView<T, 2> values) {
            auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < local_index_size) {
                for(auto level = 0; level < levels; ++level) {
                    values(local_index[idx], level) = buffer[idx * levels + level];
                }
            }
        }

        /** @brief Atlas data descriptor (GPU specialization)*/
        template <typename DomainId, typename T>
        class atlas_data_descriptor<gridtools::ghex::gpu, DomainId, T> {

            public:

                using arch_type = gridtools::ghex::gpu;
                using domain_id_type = DomainId;
                using value_type = T;
                using domain_descriptor_type = atlas_domain_descriptor<domain_id_type>;
                using index_t = typename domain_descriptor_type::index_t;
                using device_id_type = gridtools::ghex::arch_traits<arch_type>::device_id_type;

            private:

                domain_id_type m_domain_id;
                device_id_type m_device_id;
                atlas::array::ArrayView<value_type, 2> m_values;

            public:

                /** @brief constructs a GPU data descriptor
                 * @param domain local domain instance
                 * @param device_id device id
                 * @param field Atlas field to be wrapped*/
                atlas_data_descriptor(
                        const domain_descriptor_type& domain,
                        const device_id_type device_id,
                        const atlas::Field& field) :
                    m_domain_id{domain.domain_id()},
                    m_device_id{device_id},
                    m_values{atlas::array::make_device_view<value_type, 2>(field)} {}

                /** @brief data type size, mandatory*/
                std::size_t data_type_size() const {
                    return sizeof (value_type);
                }

                domain_id_type domain_id() const { return m_domain_id; }

                device_id_type device_id() const { return m_device_id; };

                template<typename IndexContainer>
                void pack(value_type* buffer, const IndexContainer& c, void* stream_ptr) {
                    for (const auto& is : c) {
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_index().size()) / GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
                        pack_kernel<value_type, index_t><<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                                m_values,
                                is.local_index().size(),
                                &(is.local_index()[0]),
                                is.levels(),
                                is.size(),
                                buffer);
                    }
                }

                template<typename IndexContainer>
                void unpack(const value_type* buffer, const IndexContainer& c, void* stream_ptr) {
                    for (const auto& is : c) {
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_index().size()) / GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
                        unpack_kernel<value_type, index_t><<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                                is.size(),
                                buffer,
                                is.local_index().size(),
                                &(is.local_index()[0]),
                                is.levels(),
                                m_values);
                    }
                }

        };

#endif

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP */
