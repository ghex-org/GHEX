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
#ifndef INCLUDED_PATTERN_HPP
#define INCLUDED_PATTERN_HPP

#include "./protocol/setup.hpp"
#include "./protocol/mpi.hpp"
#include "./buffer_info.hpp"

namespace gridtools {

    namespace detail {
        // foward declaration
        template<typename GridType>
        struct make_pattern_impl;
    } // namespace detail

    // forward declaration
    template<typename P, typename GridType, typename DomainIdType>
    class pattern;

    /** @brief an iterable holding communication patterns (one pattern per domain)
     * @tparam P transport protocol
     * @tparam GridType indicates structured/unstructured grids
     * @tparam DomainIdType type to uniquely identify partail (local) domains*/
    template<typename P, typename GridType, typename DomainIdType>
    class pattern_container
    {
    public: // member tyes
        /** @brief pattern type this object is holding */
        using value_type = pattern<P,GridType,DomainIdType>;

    private: // private member types
        using data_type  = std::vector<value_type>;

    private: // friend declarations
        friend class detail::make_pattern_impl<GridType>;

    public: // copy constructor
        pattern_container(pattern_container&&) noexcept = default;

    private: // private constructor called through make_pattern
        //pattern_container(data_type&& d) noexcept : m_patterns(d) {}
        pattern_container(data_type&& d, int mt) noexcept : m_patterns(d), m_max_tag(mt) {}

    public: // member functions
        int size() const noexcept { return m_patterns.size(); }
        const auto& operator[](int i) const noexcept { return m_patterns[i]; }
        auto begin() const noexcept { return m_patterns.cbegin(); }
        auto end() const noexcept { return m_patterns.cend(); }
        int max_tag() const noexcept { return m_max_tag; }

        /** @brief bind a field to a pattern
         * @tparam Field field type
         * @param field field instance
         * @return lightweight buffer_info object. Attention: holds references to field and pattern! */
        template<typename Field>
        buffer_info<value_type,typename Field::device_type,Field> operator()(Field& field) const
        {
            // linear search here
            for (auto& p : m_patterns)
                if (p.domain_id()==field.domain_id()) return p(*this,field);
            throw std::runtime_error("field incompatible with available domains!");
        }

    private: // members
        data_type m_patterns;
        int m_max_tag;
    };

    namespace detail {
        // implementation detail
        template<typename GridType, typename P, typename HaloGenerator, typename DomainRange>
        auto make_pattern(protocol::setup_communicator& setup_comm, protocol::communicator<P>& comm, HaloGenerator&& hgen, DomainRange&& d_range)
        {
            using grid_type = typename GridType::template type<typename std::remove_reference_t<DomainRange>::value_type>;
            return detail::make_pattern_impl<grid_type>::apply(setup_comm, comm, std::forward<HaloGenerator>(hgen), std::forward<DomainRange>(d_range)); 
        }
    } // namespace detail

    // helper function if transport protocol is also MPI
    template<typename GridType, typename HaloGenerator, typename DomainRange>
    auto make_pattern(MPI_Comm mpi_comm, HaloGenerator&& hgen, DomainRange&& d_range)
    {
        protocol::communicator<protocol::mpi> mpi_comm_(mpi_comm);
        protocol::setup_communicator setup_comm(mpi_comm);
        return detail::make_pattern<GridType>(setup_comm, mpi_comm_, hgen, d_range);
    }

    /**
     * @brief construct a pattern for each domain and establish neighbor relationships
     * @tparam GridType indicates structured/unstructured grids
     * @tparam P transport protocol
     * @tparam HaloGenerator function object which takes a domain as argument
     * @tparam DomainRange a range type holding domains
     * @param mpi_comm MPI communicator (used for establishing network topology)
     * @param comm custom communicator used in the actual exchange operations
     * @param hgen receive halo generator function object (emits iteration spaces (global coordinates) or index lists (global indices)
     * @param d_range range of local domains
     * @return iterable of patterns (one per domain) 
     */
    template<typename GridType, typename P, typename HaloGenerator, typename DomainRange>
    auto make_pattern(MPI_Comm mpi_comm, protocol::communicator<P>& comm, HaloGenerator&& hgen, DomainRange&& d_range)
    {
        protocol::setup_communicator setup_comm(mpi_comm);
        return detail::make_pattern<GridType>(setup_comm, comm, hgen, d_range);
    }

} // namespace gridtools

#endif /* INCLUDED_PATTERN_HPP */

