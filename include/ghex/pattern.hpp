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
#ifndef INCLUDED_GHEX_PATTERN_HPP
#define INCLUDED_GHEX_PATTERN_HPP

#include "./transport_layer/mpi/setup.hpp"
#include "./transport_layer/mpi/communicator.hpp"
#include "./halo.hpp"

namespace gridtools {

    namespace ghex {

        namespace detail {
            // forward declaration
            template<typename GridType>
            struct make_pattern_impl;
        } // namespace detail

        // forward declaration
        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern;

        /** @brief an iterable holding communication patterns (one pattern per domain)
         * @tparam Transport transport protocol
         * @tparam GridType indicates structured/unstructured grids
         * @tparam DomainIdType type to uniquely identify partail (local) domains*/
        template<typename Transport, typename GridType, typename DomainIdType>
        class pattern_container
        {
        public: // member tyes
            /** @brief pattern type this object is holding */
            using value_type = pattern<Transport,GridType,DomainIdType>;
            template<typename Field>
            using halo_type  = halo_t<value_type,Field>;

        private: // private member types
            using data_type  = std::vector<value_type>;

        private: // friend declarations
            friend class detail::make_pattern_impl<GridType>;

        public: // copy constructor
            pattern_container(const pattern_container&) noexcept = delete;
            pattern_container(pattern_container&&) noexcept = default;

        private: // private constructor called through make_pattern
            pattern_container(data_type&& d, int mt) noexcept : m_patterns(d), m_max_tag(mt) 
            {
                for (auto& p : m_patterns)
                    p.m_container = this;
            }

        public: // member functions
            int size() const noexcept { return m_patterns.size(); }
            const auto& operator[](int i) const noexcept { return m_patterns[i]; }
            auto begin() const noexcept { return m_patterns.cbegin(); }
            auto end() const noexcept { return m_patterns.cend(); }
            int max_tag() const noexcept { return m_max_tag; }

            /** @brief generates halo exchange object for a given field.
              * precondition: Field has a member function domain_id() which returns the id of the associated domain. 
              * @tparam Field data descriptor type
              * @param field data field
              * @return halo exchange object*/
            template<typename Field>
            halo_type<Field> halo(Field& field) const
            {
                // linear search here
                for (const auto& p : m_patterns)
                    if (p.domain_id() == field.domain_id()) return { p, field.domain_id(), field};
                throw std::runtime_error("Domain incompatible with pattern!");
            }

            /** @brief generates halo exchange object for a given domain and field.
              * @tparam Field data descriptor type
              * @tparam Domain domain type
              * @param domain domain instance
              * @param field data field
              * @return halo exchange object*/
            template<typename Field, typename Domain>
            halo_type<Field> halo(const Domain& domain, Field& field) const
            {
                // linear search here
                for (const auto& p : m_patterns)
                    if (p.domain_id() == domain.domain_id()) return { p, p.domain_id(), field};
                throw std::runtime_error("Domain incompatible with pattern!");
            }

        private: // members
            data_type m_patterns;
            int m_max_tag;
        };

        namespace detail {
            // implementation detail
            template<typename GridType, typename Transport, typename HaloGenerator, typename DomainRange>
            auto make_pattern(tl::mpi::setup_communicator& setup_comm, tl::communicator<Transport>& comm, HaloGenerator&& hgen, DomainRange&& d_range)
            {
                using grid_type = typename GridType::template type<typename std::remove_reference_t<DomainRange>::value_type>;
                return detail::make_pattern_impl<grid_type>::apply(setup_comm, comm, std::forward<HaloGenerator>(hgen), std::forward<DomainRange>(d_range)); 
            }
        } // namespace detail

        // helper function if transport protocol is also MPI
        template<typename GridType, typename HaloGenerator, typename DomainRange>
        auto make_pattern(MPI_Comm mpi_comm, HaloGenerator&& hgen, DomainRange&& d_range)
        {
            tl::communicator<tl::mpi_tag> mpi_comm_{mpi_comm};
            tl::mpi::setup_communicator setup_comm(mpi_comm);
            return detail::make_pattern<GridType>(setup_comm, mpi_comm_, hgen, d_range);
        }

        /**
         * @brief construct a pattern for each domain and establish neighbor relationships
         * @tparam GridType indicates structured/unstructured grids
         * @tparam Transport transport protocol
         * @tparam HaloGenerator function object which takes a domain as argument
         * @tparam DomainRange a range type holding domains
         * @param mpi_comm MPI communicator (used for establishing network topology)
         * @param comm custom communicator used in the actual exchange operations
         * @param hgen receive halo generator function object (emits iteration spaces (global coordinates) or index lists (global indices)
         * @param d_range range of local domains
         * @return iterable of patterns (one per domain) 
         */
        template<typename GridType, typename Transport, typename HaloGenerator, typename DomainRange>
        auto make_pattern(MPI_Comm mpi_comm, tl::communicator<Transport>& comm, HaloGenerator&& hgen, DomainRange&& d_range)
        {
            tl::mpi::setup_communicator setup_comm(mpi_comm);
            return detail::make_pattern<GridType>(setup_comm, comm, hgen, d_range);
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_PATTERN_HPP */

