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

#include "protocol/setup.hpp"
#include "protocol/mpi.hpp"
#include <vector>

namespace gridtools {

    /** @brief generic communication pattern
     * @tparam P transport protocol
     * @tparam GridType indicates structured/unstructured grids
     * @tparam DomainIdType type to uniquely identify partial (local) domains*/
    template<typename P, typename GridType, typename DomainIdType>
    class pattern
    {};

    /** @brief an iterable holding communication patterns (one pattern per domain)
     * @tparam P transport protocol
     * @tparam GridType indicates structured/unstructured grids
     * @tparam DomainIdType type to uniquely identify partail (local) domains*/
    template<typename P, typename GridType, typename DomainIdType>
    struct pattern_container
    {
        using value_type = pattern<P,GridType,DomainIdType>;
        using data_type  = std::vector<value_type>;

        pattern_container(data_type&& d) noexcept : m_patterns(d) {}
        pattern_container(pattern_container&&) noexcept = default;

        int size() const noexcept { return m_patterns.size(); }

        const auto& operator[](int i) const noexcept { return m_patterns[i]; }

        auto begin() const noexcept { return m_patterns.cbegin(); }
        auto end() const noexcept { return m_patterns.cend(); }
    private:
        std::vector<pattern<P,GridType,DomainIdType>> m_patterns;
    };

    namespace detail {

        template<typename GridType>
        struct make_pattern_impl {};

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
     * @return 
     */
    template<typename GridType, typename P, typename HaloGenerator, typename DomainRange>
    auto make_pattern(MPI_Comm mpi_comm, protocol::communicator<P>& comm, HaloGenerator&& hgen, DomainRange&& d_range)
    {
        protocol::setup_communicator setup_comm(mpi_comm);
        return detail::make_pattern<GridType>(setup_comm, comm, hgen, d_range);
    }

} // namespace gridtools

#endif /* INCLUDED_PATTERN_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

