/*
 * GridTools
 *
 * Copyright (c) 2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#ifndef GHEX_MPI_COMMUNICATOR_TRAITS_HPP
#define GHEX_MPI_COMMUNICATOR_TRAITS_HPP

#include <mpi.h>

namespace gridtools
{
namespace ghex
{
namespace mpi
{

class communicator_traits
{
    MPI_Comm m_comm;

public:
    communicator_traits(MPI_Comm comm)
        : m_comm{comm}
    {
    }

    communicator_traits()
        : m_comm{MPI_COMM_WORLD}
    {
    }

    MPI_Comm communicator() const { return m_comm; }
};

} // namespace mpi
} // namespace ghex
} // namespace gridtools

#endif