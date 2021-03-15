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
         * An Atlas domain is assumed to include the halo region as well,
         * and has therefore to be istantiated using a mesh which has already grown the required halo layer
         * after the creation of a function space with a halo.
         * Null halo is fine too, provided that the mesh is in its final state.
         * Domain size includes halo size.
         * @tparam DomainId domain id type*/
        template<typename DomainId>
        class atlas_domain_descriptor {

            public:

                // member types
                using domain_id_type = DomainId;
                using local_index_type = atlas::idx_t;

            private:

                // members
                domain_id_type m_id;
                atlas::Field m_partition;
                atlas::Field m_remote_index;
                local_index_type m_size;
                std::size_t m_levels;

            public:

                // ctors
                /** @brief Constructs a local domain
                 * @param id domain id
                 * @param partition partition indices of domain (+ halo) elements (Atlas field)
                 * @param remote_index local indices in remote partition for domain (+ halo) elements (Atlas field)
                 * @param size number of domain + halo points*/
                atlas_domain_descriptor(const domain_id_type id,
                                        const atlas::Field& partition,
                                        const atlas::Field& remote_index,
                                        const std::size_t levels) :
                    m_id{id},
                    m_partition{partition},
                    m_remote_index{remote_index},
                    m_size{partition.size()},
                    m_levels{levels} {

                    // Asserts
                    assert(partition.size() == remote_index.size());

                }

                // member functions
                domain_id_type domain_id() const noexcept { return m_id; }
                const atlas::Field& partition() const noexcept { return m_partition; }
                const atlas::Field& remote_index() const noexcept { return m_remote_index; }
                local_index_type size() const noexcept { return m_size; }
                std::size_t levels() const noexcept { return m_levels; }

                // print
                /** @brief print */
                template<class CharT, class Traits>
                friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const atlas_domain_descriptor& domain) {
                    os << "domain id = " << domain.domain_id() << ";\n"
                       << "size = " << domain.size() << ";\n"
                       << "levels = " << domain.levels() << ";\n"
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
        class atlas_halo_generator {

            public:

                // member types
                using domain_type = atlas_domain_descriptor<DomainId>;
                using local_index_type = typename domain_type::local_index_type;

                /** @brief Halo class for Atlas
                  * Provides list of local indices of neighboring elements.*/
                class halo {

                    private:

                        std::vector<local_index_type> m_local_indices;
                        std::size_t m_levels;

                    public:

                        // ctors
                        halo(const std::size_t levels) : m_levels{levels} {}

                        // member functions
                        std::size_t size() const noexcept { return m_local_indices.size(); }
                        std::size_t levels() const noexcept { return m_levels; }
                        std::vector<local_index_type>& local_indices() noexcept { return m_local_indices; }
                        const std::vector<local_index_type>& local_indices() const noexcept { return m_local_indices; }

                        // print
                        /** @brief print */
                        template<class CharT, class Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const halo& h) {
                            os << "size = " << h.size() << ";\n"
                               << "levels = " << h.levels() << ";\n"
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
                halo operator()(const domain_type& domain) const {

                    auto partition = atlas::array::make_view<int, 1>(domain.partition());
                    auto remote_index = atlas::array::make_view<local_index_type, 1>(domain.remote_index());

                    halo h{domain.levels()};

                    // if the index refers to another domain, or even to the same but as a halo point,
                    // the halo is updated
                    for (local_index_type d_idx = 0; d_idx < domain.size(); ++d_idx) {
                        if ((partition(d_idx) != domain.domain_id()) || (remote_index(d_idx) != d_idx)) {
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
        class atlas_recv_domain_ids_gen {

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
                class halo {

                    private:

                        std::vector<domain_id_type> m_domain_ids;
                        std::vector<local_index_type> m_remote_indices;
                        std::vector<int> m_ranks;

                    public:

                        // member functions
                        std::vector<domain_id_type>& domain_ids() noexcept { return m_domain_ids; }
                        const std::vector<domain_id_type>& domain_ids() const noexcept { return m_domain_ids; }
                        std::vector<local_index_type>& remote_indices() noexcept { return m_remote_indices; }
                        const std::vector<local_index_type>& remote_indices() const noexcept { return m_remote_indices; }
                        std::vector<int>& ranks() noexcept { return m_ranks; }
                        const std::vector<int>& ranks() const noexcept { return m_ranks; }

                        // print
                        /** @brief print */
                        template<class CharT, class Traits>
                        friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const halo& h) {
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
                halo operator()(const domain_type& domain) const {

                    auto partition = atlas::array::make_view<int, 1>(domain.partition());
                    auto remote_index = atlas::array::make_view<local_index_type, 1>(domain.remote_index());

                    halo h{};

                    // if the index refers to another domain, or even to the same but as a halo point,
                    // the halo is updated
                    for (local_index_type d_idx = 0; d_idx < domain.size(); ++d_idx) {
                        if ((partition(d_idx) != domain.domain_id()) || (remote_index(d_idx) != d_idx)) {
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
                using local_index_type = typename domain_descriptor_type::local_index_type;
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

                domain_id_type domain_id() const { return m_domain_id; }

                device_id_type device_id() const { return 0; }

                int num_components() const noexcept { return 1; }

                /** @brief single access operator, used by multiple access set function*/
                value_type& operator()(const local_index_type idx, const local_index_type level) {
                    return m_values(idx, level);
                }

                /** @brief single access operator (const version), used by multiple access get function*/
                const value_type& operator()(const local_index_type idx, const local_index_type level) const {
                    return m_values(idx, level);
                }

                /** @brief multiple access set function, needed by GHEX to perform the unpacking
                 * @tparam IterationSpace iteration space type
                 * @param is iteration space which to loop through when setting back the buffer values
                 * @param buffer buffer with the data to be set back*/
                template <typename IterationSpace>
                void set(const IterationSpace& is, const byte_t* buffer) {
                    for (local_index_type idx : is.local_indices()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            std::memcpy(&((*this)(idx, level)), buffer, sizeof(value_type)); // level: implicit cast to local_index_type
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
                    for (local_index_type idx : is.local_indices()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            std::memcpy(buffer, &((*this)(idx, level)), sizeof(value_type)); // level: implicit cast to local_index_type
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

        template <typename T, typename local_index_type>
        __global__ void pack_kernel(
                const atlas::array::ArrayView<T, 2> values,
                const std::size_t local_indices_size,
                const local_index_type* local_indices,
                const std::size_t levels,
                T* buffer) {
            auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < local_indices_size) {
                for(auto level = 0; level < levels; ++level) {
                    buffer[idx * levels + level] = values(local_indices[idx], level);
                }
            }
        }

        template <typename T, typename local_index_type>
        __global__ void unpack_kernel(
                const T* buffer,
                const std::size_t local_indices_size,
                const local_index_type* local_indices,
                const std::size_t levels,
                atlas::array::ArrayView<T, 2> values) {
            auto idx = threadIdx.x + (blockIdx.x * blockDim.x);
            if (idx < local_indices_size) {
                for(auto level = 0; level < levels; ++level) {
                    values(local_indices[idx], level) = buffer[idx * levels + level];
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
                using local_index_type = typename domain_descriptor_type::local_index_type;
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
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) / GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
                        pack_kernel<value_type, local_index_type><<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                                m_values,
                                is.local_indices().size(),
                                &(is.local_indices()[0]),
                                is.levels(),
                                buffer);
                    }
                }

                template<typename IndexContainer>
                void unpack(const value_type* buffer, const IndexContainer& c, void* stream_ptr) {
                    for (const auto& is : c) {
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_indices().size()) / GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK));
                        unpack_kernel<value_type, local_index_type><<<n_blocks, GHEX_ATLAS_SERIALIZATION_THREADS_PER_BLOCK, 0, *(reinterpret_cast<cudaStream_t*>(stream_ptr))>>>(
                                buffer,
                                is.local_indices().size(),
                                &(is.local_indices()[0]),
                                is.levels(),
                                m_values);
                    }
                }

        };

#endif

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_GLUE_ATLAS_USER_CONCEPTS_HPP */
