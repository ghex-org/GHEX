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
#ifndef INCLUDED_GHEX_UNSTRUCTURED_USER_CONCEPTS_HPP
#define INCLUDED_GHEX_UNSTRUCTURED_USER_CONCEPTS_HPP

#include <set>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <cstring>
#include <iosfwd>

#include "./grid.hpp"
#include "../arch_list.hpp"
#include "../arch_traits.hpp"
#include "../allocator/unified_memory_allocator.hpp"

#ifdef __CUDACC__
#include "../../cuda_utils/error.hpp"
#endif


namespace gridtools {

    namespace ghex {

        namespace unstructured {

            /** @brief domain descriptor for unstructured domains
             * @tparam DomainId domain id type
             * @tparam Idx global index type*/
            template<typename DomainId, typename Idx>
            class domain_descriptor {

                public:

                    // member types
                    using domain_id_type = DomainId;
                    using global_index_type = Idx;
                    using index_type = std::size_t; // TO DO: should be derived? (check better)
                    using vertices_type = std::set<global_index_type>;
                    using xadj_type = std::vector<index_type>; // named after ParMetis CSR arrays
                    using adjncy_type = std::vector<global_index_type>; // named after ParMetis CSR arrays
                    using map_type = std::vector<std::pair<global_index_type, adjncy_type>>;

                private:

                    // members
                    domain_id_type m_id;
                    vertices_type m_vertices;
                    xadj_type m_xadj; // named after ParMetis CSR arrays
                    adjncy_type m_adjncy; // named after ParMetis CSR arrays
                    vertices_type m_halo_vertices;

                public:

                    // constructors
                    domain_descriptor() = default;
                    domain_descriptor(const domain_id_type id,
                                      const vertices_type& vertices,
                                      const xadj_type& xadj,
                                      const adjncy_type& adjncy) :
                        m_id{id},
                        m_vertices{vertices},
                        m_xadj{xadj},
                        m_adjncy{adjncy},
                        m_halo_vertices{} { set_halo_vertices(); }
                    domain_descriptor(const domain_id_type id,
                                      const map_type& v_map) :
                        m_id{id},
                        m_vertices{},
                        m_xadj{static_cast<index_type>(0)}, // first element is 0 by construction
                        m_adjncy{},
                        m_halo_vertices{} {
                        for (const auto& v_elem : v_map) {
                            m_vertices.insert(v_elem.first);
                            m_xadj.push_back(static_cast<index_type>(v_elem.second.size()));
                            m_adjncy.insert(m_adjncy.end(), v_elem.second.begin(), v_elem.second.end());
                            set_halo_vertices();
                        }
                    }

                private:

                    // member functions
                    void set_halo_vertices() {
                        vertices_type all_vertices{};
                        all_vertices.insert(m_adjncy().begin(), m_adjncy().end());
                        std::set_difference(all_vertices.begin(), all_vertices.end(),
                                            m_vertices().begin(), m_vertices().end(),
                                            std::inserter(m_halo_vertices, m_halo_vertices.begin()));
                    }

                public:

                    // member functions
                    domain_id_type domain_id() const noexcept { return m_id; }
                    /** @brief domain size, not including halo */
                    std::size_t size() const noexcept { return m_vertices.size(); }
                    const vertices_type& vertices() const noexcept { return m_vertices; }
                    const xadj_type& xadj() const noexcept { return m_xadj; }
                    const adjncy_type& adjncy() const noexcept { return m_adjncy; }
                    const vertices_type& halo_vertices() const noexcept { return m_halo_vertices; }

                    // print
                    /** @brief print */
                    template<typename CharT, typename Traits>
                    friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const domain_descriptor& domain) {
                        os << "domain id = " << domain.domain_id() << ";\n"
                           << "size = " << domain.size() << ";\n"
                           << "vertices: [ ";
                        for (auto v : domain.vertices()) { os << v << " "; }
                        os << "]\n";
                        os << "xadj: [ ";
                        for (auto x : domain.xadj()) { os << x << " "; }
                        os << "]\n";
                        os << "adjncy: [ ";
                        for (auto v : domain.adjncy()) { os << v << " "; }
                        os << "]\n";
                        return os;
                    }

            };

            /** @brief halo generator for unstructured domains
             * @tparam DomainId domain id type
             * @tparam Idx global index type*/
            template<typename DomainId, typename Idx>
            class halo_generator {

                public:

                    // member types
                    using domain_type = domain_descriptor<DomainId, Idx>;
                    using global_index_type = typename domain_type::global_index_type;
                    using vertices_type = typename domain_type::vertices_type;

                    /** @brief Halo concept for unstructured grids
                     * TO DO: if everything works, this class definition should be removed,
                     * iteration space concept should be moved outside the pattern class,
                     * templated on the index type and used here as well.*/
                    class halo {

                        private:

                            vertices_type m_vertices;

                        public:

                            // ctors
                            halo() noexcept = default;
                            /** WARN: following one not strictly needed,
                             * but it will if this class is used as iteration_space class*/
                            halo(const vertices_type& vertices) : m_vertices{vertices} {}

                            // member functions
                            /** @brief size of the halo */
                            std::size_t size() const noexcept { return m_vertices.size(); }
                            vertices_type& vertices() noexcept { return m_vertices; }
                            const vertices_type& vertices() const noexcept { return m_vertices; }
                            void push_back(const global_index_type v) { m_vertices.push_back(v); }

                            // print
                            /** @brief print */
                            template<typename CharT, typename Traits>
                            friend std::basic_ostream<CharT, Traits>& operator << (std::basic_ostream<CharT, Traits>& os, const halo& h) {
                                os << "size = " << h.size() << ";\n"
                                   << "vertices: [ ";
                                for (auto v : h.vertices()) { os << v << " "; }
                                os << "]\n";
                                return os;
                            }

                    };

                    // member functions
                    /** @brief generate halo(s)
                     * @param domain local domain instance
                     * @return (vector of) receive halo(s)
                     * TO DO: now it generates only one halo: should this actually be the rule?
                     * Should make_pattern be modified accordingly?*/
                    auto operator()(const domain_type& domain) const {
                        std::vector<halo> halos{};
                        halos.push_back({domain.halo_vertices()});
                        return halos;
                    }

            };

            /** @brief data descriptor for unstructured grids (forward declaration)
             * @tparam Arch device type in which field storage is allocated
             * @tparam DomainId domain id type
             * @tparam Idx global index type
             * @tparam T value type*/
            template <typename Arch, typename DomainId, typename Idx, typename T>
            class data_descriptor;

            /** @brief data descriptor for unstructured grids (CPU specialization)*/
            template <typename DomainId, typename Idx, typename T>
            class data_descriptor<gridtools::ghex::cpu, DomainId, Idx, T> {

                public:

                    using arch_type = gridtools::ghex::cpu;
                    using domain_id_type = DomainId;
                    using global_index_type = Idx;
                    using value_type = T;
                    using device_id_type = gridtools::ghex::arch_traits<arch_type>::device_id_type;
                    using index_type = std::size_t; // TO DO: should be the same of the Iteration space (check better)
                    using domain_descriptor_type = domain_descriptor<domain_id_type, global_index_type>;
                    using storage_type = std::vector<T>;
                    using byte_t = unsigned char;

                private:

                    domain_id_type m_domain_id;
                    storage_type m_values;
                    index_type m_levels; // TO DO: make it more abstract, should be one for each structured dimension

                public:

                    // constructors
                    /** @brief constructs a CPU data descriptor
                     * @tparam Container templated container type for the field to be wrapped
                     * @param domain local domain instance
                     * @param field field to be wrapped
                     * @param levels number of vertical layers for semi-structured grids*/
                    template <template <typename> typename Container>
                    data_descriptor(const domain_descriptor_type& domain,
                                    const Container<value_type>& field,
                                    const index_type levels = 1) :
                        m_domain_id{domain.domain_id()},
                        m_values{field.begin(), field.end()},
                        m_levels {levels} {
                        assert(field.size() == ((domain.size() + domain.halo_vertices().size()) * levels));
                    }

                    // member functions

                    std::size_t data_type_size() const noexcept { return sizeof (value_type); } // TO DO: really needed?
                    domain_id_type domain_id() const noexcept { return m_domain_id; }
                    device_id_type device_id() const noexcept { return 0; } // significant for the GPU
                    index_type levels() const noexcept { return m_levels; }

                    /** @brief single access operator, used by multiple access set function*/
                    value_type& operator()(const index_type local_v, const index_type level) {
                        return m_values[local_v * m_levels + level];
                    }

                    /** @brief single access operator (const version), used by multiple access get function*/
                    const value_type& operator()(const index_type local_v, const index_type level) const {
                        return m_values[local_v * m_levels + level];
                    }

                    /** @brief multiple access set function, needed by GHEX to perform the unpacking
                     * @tparam IterationSpace iteration space type
                     * @param is iteration space which to loop through when setting back the buffer values
                     * @param buffer buffer with the data to be set back*/
                    template <typename IterationSpace>
                    void set(const IterationSpace& is, const byte_t* buffer) {
                        for (index_type local_v : is.local_index()) /* TO DO: change names in iteration_space */ {
                            for (index_type level = 0; level < is.levels(); ++level) {
                                std::memcpy(&((*this)(local_v, level)), buffer, sizeof(value_type));
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
                        for (index_type local_v : is.local_index()) /* TO DO: change names in iteration_space */ {
                            for (index_type level = 0; level < is.levels(); ++level) {
                                std::memcpy(buffer, &((*this)(local_v, level)), sizeof(value_type));
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
        // TO DO: GPU SPECIALIZATION
#endif

        } // namespace unstructured

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_UNSTRUCTURED_USER_CONCEPTS_HPP */
