/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <ghex/context.hpp>

namespace ghex
{
context::context(MPI_Comm comm, bool thread_safe)
: m_ctxt{std::make_unique<oomph::context>(comm, thread_safe)}
{
    GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(mpi_comm(), &m_rank));
    GHEX_CHECK_MPI_RESULT(MPI_Comm_size(mpi_comm(), &m_size));
}

context::message_type
context::make_buffer(std::size_t size)
{
    return m_ctxt->template make_buffer<unsigned char>(size);
}

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
context::message_type
context::make_device_buffer(std::size_t size, int id)
{
    return m_ctxt->template make_device_buffer<unsigned char>(size, id);
}
#endif

} // namespace ghex
