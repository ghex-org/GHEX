/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/mpi/error.hpp>
#include <oomph/context.hpp>
#include <memory>

namespace ghex
{
class barrier;
class context
{
    friend class barrier;
  public:
    using rank_type = oomph::rank_type;
    using tag_type = oomph::tag_type;
    using communicator_type = oomph::communicator;
    using message_type = oomph::message_buffer<unsigned char>;

  private:
    std::unique_ptr<oomph::context> m_ctxt;
    rank_type                       m_rank;
    rank_type                       m_size;

  public:
    context(MPI_Comm comm, bool thread_safe = true);

  public:
    auto      transport_context() const noexcept { return m_ctxt.get(); }
    MPI_Comm  mpi_comm() const noexcept { return m_ctxt->mpi_comm(); }
    rank_type rank() const noexcept { return m_rank; }
    rank_type size() const noexcept { return m_size; }

    communicator_type get_communicator() { return m_ctxt->get_communicator(); }

    message_type make_buffer(std::size_t size);

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    message_type make_device_buffer(std::size_t size, int id);
#endif
};

} // namespace ghex
