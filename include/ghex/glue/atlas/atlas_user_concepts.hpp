/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
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

        // forward declaration
        template<typename DomainId>
        class atlas_halo_generator;

        /** @brief implements domain descriptor concept for Atlas domains
         * An Atlas domain has inner notion of the halos.
         * It has to be istantiated using a mesh which has already grown the required halo layer
         * due to the creation of a function space with a halo.
         * Null halo is fine too, provided that the mesh is in its final state.
         * Be careful: domain size includes halos!
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
                std::size_t m_levels; // WARN: should it be of index_t ?
                index_t m_size;
                index_t m_first;
                index_t m_last;

            public:

                // ctors
                // NOTE: a constructor which takes as argument a reference to the whole mesh
                // may not be a good idea, since it has to be spoecialized for a given function space.
                /** @brief construct a local domain
                 * @param id domain id
                 * @param partition field with partition indexes
                 * @param remote_index field with local indexes in remote partition
                 * @param size size of the domain + all required halo points
                 * @param levels number of vertical levels
                 * @param rank PE rank*/
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
                       << "partition indexes: [" << domain.partition() << "]\n"
                       << "remote indexes: [" << domain.remote_index() << "]\n";
                    return os;
                }

        };

        /** @brief halo generator for atlas domains
         * An Atlas domain has already the notion of halos.
         * The purpose of the Atlas halo generator is to fill a container
         * with indexes referring to remote partitions.
         * @tparam DomainId domain id type*/
        template<typename DomainId>
        class atlas_halo_generator {

            public:

                // member types
                using domain_type = atlas_domain_descriptor<DomainId>;
                using index_t = atlas::idx_t;

                /** @brief essentially a partition index and a sequence of remote indexes;
                 * conceptually, it is similar to an iteration space
                 * (only differences are that here it is already specialized for atlas
                 * and provides an insert method as well),
                 * and this is why the two concepts could potentially coincide.
                 * WARN: size() gives only horizontal size.*/
                class halo {

                    private:

                        int m_partition;
                        std::vector<index_t, gridtools::ghex::allocator::unified_memory_allocator<index_t>> m_local_index;
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
                        halo(const int partition,
                             const std::vector<index_t, gridtools::ghex::allocator::unified_memory_allocator<index_t>>& local_index,
                             const std::vector<index_t>& remote_index,
                             const std::size_t levels = 1) noexcept :
                            m_partition{partition},
                            m_local_index{local_index},
                            m_remote_index{remote_index},
                            m_levels{levels} {
                            assert(local_index.size() == remote_index.size());
                        } // WARN: is this constructor actually needed?

                        // member functions
                        int partition() const noexcept { return m_partition; }
                        std::vector<index_t, gridtools::ghex::allocator::unified_memory_allocator<index_t>>& local_index() noexcept { return m_local_index; }
                        const std::vector<index_t, gridtools::ghex::allocator::unified_memory_allocator<index_t>>& local_index() const noexcept { return m_local_index; }
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
                               << "local indexes: [ ";
                            for (auto idx : h.local_index()) { os << idx << " "; }
                            os << "]\nremote indexes: [ ";
                            for (auto idx : h.remote_index()) { os << idx << " "; }
                            os << "]\n";
                            return os;
                        }

                };

            private:

                //members
                int m_rank;
                int m_size;

            public:

                // ctors
                /** @brief construct a halo generator
                 * @param rank PE rank
                 * @param size number of PEs*/
                atlas_halo_generator(const int rank, const int size) noexcept :
                    m_rank{rank},
                    m_size{size} {}

                // member functions
                /** @brief generate halos
                 * @param domain local domain instance
                 * @return vector of receive halos, one for each PE*/
                auto operator()(const domain_type& domain) const {

                    std::vector<halo> halos{};
                    for (int rank = 0; rank < m_size; ++rank) {
                        halos.push_back({rank, domain.levels()});
                    }

                    const int* partition_data = atlas::array::make_view<int, 1>(domain.partition()).data();
                    const index_t* remote_index_data = atlas::array::make_view<index_t, 1>(domain.remote_index()).data();

                    // if the index refers to another rank, or even to the same rank but as a halo point,
                    // corresponding halo is updated
                    for (auto domain_idx = 0; domain_idx < domain.size(); ++domain_idx) {
                        if ((partition_data[domain_idx] != m_rank) || (remote_index_data[domain_idx] != domain_idx)) {
                            halos[partition_data[domain_idx]].push_back(domain_idx, remote_index_data[domain_idx]);
                        }
                    }

                    return halos;

                }

        };

        /** @brief CPU data descriptor*/
        template <typename T, typename DomainDescriptor>
        class atlas_data_descriptor {

            public:

                using value_type = T;
                using index_t = typename DomainDescriptor::index_t;
                using domain_id_t = typename DomainDescriptor::domain_id_type;
                using arch_type = gridtools::ghex::cpu;
                using device_id_type = gridtools::ghex::arch_traits<arch_type>::device_id_type;
                using Byte = unsigned char;

            private:

                const DomainDescriptor& m_domain;
                atlas::array::ArrayView<T, 2> m_values;

            public:

                atlas_data_descriptor(const DomainDescriptor& domain,
                                      const atlas::Field& field) :
                    m_domain{domain},
                    m_values{atlas::array::make_view<T, 2>(field)} {}

                /** @brief data type size, mandatory*/
                std::size_t data_type_size() const {
                    return sizeof (T);
                }

                domain_id_t domain_id() const { return m_domain.domain_id(); }

                device_id_type device_id() const { return 0; }

                /** @brief single access set function, not mandatory but used by the corresponding multiple access operator*/
                void set(const T& value, const index_t idx, const std::size_t level) {
                    m_values(static_cast<std::size_t>(idx), level) = value; // WARN: why std::size_t cast is needed here?
                }

                /** @brief single access get function, not mandatory but used by the corresponding multiple access operator*/
                const T& get(const index_t idx, const std::size_t level) const {
                    return m_values(static_cast<std::size_t>(idx), level); // WARN: why std::size_t cast is needed here?
                }

                /** @brief multiple access set function, needed by GHEX in order to perform the unpacking.
                 * WARN: it could be more efficient if the iteration space includes also the indexes on this domain;
                 * in order to do so, iteration space needs to include an additional set of indexes;
                 * for now, the needed indices are retrieved by looping over the whole doamin,
                 * and filtering out all the indices by those on the desired remote partition.
                 * @tparam IterationSpace iteration space type
                 * @param is iteration space which to loop through in order to retrieve the coordinates at which to set back the buffer values
                 * @param buffer buffer with the data to be set back*/
                template <typename IterationSpace>
                void set(const IterationSpace& is, const Byte* buffer) {
                    for (index_t idx : is.local_index()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            set(*(reinterpret_cast<const T*>(buffer)), idx, level);
                            buffer += sizeof(T);
                        }
                    }
                }

                /** @brief multiple access get function, needed by GHEX in order to perform the packing
                 * @tparam IterationSpace iteration space type
                 * @param is iteration space which to loop through in order to retrieve the coordinates at which to get the data
                 * @param buffer buffer to be filled*/
                template <typename IterationSpace>
                void get(const IterationSpace& is, Byte* buffer) const {
                    for (index_t idx : is.local_index()) {
                        for (std::size_t level = 0; level < is.levels(); ++level) {
                            std::memcpy(buffer, &get(idx, level), sizeof(T));
                            buffer += sizeof(T);
                        }
                    }
                }

                template<typename IndexContainer>
                void pack(T* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        get(is, reinterpret_cast<Byte*>(buffer));
                    }
                }

                template<typename IndexContainer>
                void unpack(const T* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        set(is, reinterpret_cast<const Byte*>(buffer));
                    }
                }

        };

#ifdef __CUDACC__

#define THREADS_PER_BLOCK 32

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

        /** @brief GPU data descriptor*/
        template <typename T, typename DomainDescriptor>
        class atlas_data_descriptor_gpu {

            public:

                using value_type = T;
                using index_t = typename DomainDescriptor::index_t;
                using domain_id_t = typename DomainDescriptor::domain_id_type;
                using arch_type = gridtools::ghex::gpu;
                using device_id_type = gridtools::ghex::arch_traits<arch_type>::device_id_type;

            private:

                const DomainDescriptor& m_domain;
                device_id_type m_device_id;
                atlas::array::ArrayView<T, 2> m_values;

            public:

                atlas_data_descriptor_gpu(
                        const DomainDescriptor& domain,
                        const device_id_type device_id,
                        const atlas::Field& field) : // WARN: different from cpu data descriptor, but easy to change there
                    m_domain{domain},
                    m_device_id{device_id},
                    m_values{atlas::array::make_device_view<T, 2>(field)} {}

                /** @brief data type size, mandatory*/
                std::size_t data_type_size() const {
                    return sizeof (T);
                }

                domain_id_t domain_id() const { return m_domain.domain_id(); }

                device_id_type device_id() const { return m_device_id; };

                template<typename IndexContainer>
                void pack(T* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_index().size()) / THREADS_PER_BLOCK));
                        pack_kernel<T, index_t><<<n_blocks, THREADS_PER_BLOCK>>>(
                                m_values,
                                is.local_index().size(),
                                &(is.local_index()[0]),
                                is.levels(),
                                is.size(),
                                buffer);
                    }
                }

                template<typename IndexContainer>
                void unpack(const T* buffer, const IndexContainer& c, void*) {
                    for (const auto& is : c) {
                        int n_blocks = static_cast<int>(std::ceil(static_cast<double>(is.local_index().size()) / THREADS_PER_BLOCK));
                        unpack_kernel<T, index_t><<<n_blocks, THREADS_PER_BLOCK>>>(
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
