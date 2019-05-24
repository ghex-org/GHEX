// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
#ifndef INCLUDED_PATTERN_HPP
#define INCLUDED_PATTERN_HPP

#include "protocol/setup.hpp"
#include "protocol/mpi.hpp"

namespace gridtools {

template<typename P, typename GridType, typename DomainIdType>
class pattern
{};

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

template<typename GridType, typename HaloGenerator, typename DomainRange>
auto make_pattern(MPI_Comm mpi_comm, HaloGenerator&& hgen, DomainRange&& d_range)
{
    protocol::communicator<protocol::mpi> mpi_comm_(mpi_comm);
    protocol::setup_communicator setup_comm(mpi_comm);
    return detail::make_pattern<GridType>(setup_comm, mpi_comm_, hgen, d_range);
}

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

