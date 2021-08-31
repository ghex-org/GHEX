/* 
 * GridTools
 * 
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#pragma once

#include <ghex/config.hpp>
#include <ghex/mpi/error.hpp>
#include <oomph/context.hpp>
#include <memory>

namespace ghex
{
//template<typename Pattern, typename Arch>
//class communication_object;
//namespace mpi
//{
//class communicator;
//} // namespace mpi

class context
{
  public:
    using rank_type = oomph::communicator::rank_type;
    using tag_type = oomph::communicator::tag_type;
    using communicator_type = oomph::communicator;
    using message_type = oomph::message_buffer<unsigned char>;

  private:
    //friend class communication_object;
    //template<typename Pattern, typename Arch>
    //friend class communication_object;
    //friend class mpi::communicator;

  private:
    std::unique_ptr<oomph::context> m_ctxt;
    rank_type                       m_rank;
    rank_type                       m_size;

  public:
    context(MPI_Comm comm, bool thread_safe = true);
    //: m_ctxt{std::make_unique<oomph::context>(comm, thread_safe)}
    //{
    //    GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(mpi_comm(), &m_rank));
    //    GHEX_CHECK_MPI_RESULT(MPI_Comm_size(mpi_comm(), &m_size));
    //}

  public:
    auto      transport_context() const noexcept { return m_ctxt.get(); }
    MPI_Comm  mpi_comm() const noexcept { return m_ctxt->mpi_comm(); }
    rank_type rank() const noexcept { return m_rank; }
    rank_type size() const noexcept { return m_size; }

    communicator_type get_communicator() { return m_ctxt->get_communicator(); }

    message_type make_buffer(std::size_t size);
    //{
    //    return m_ctxt->template make_buffer<unsigned char>(size);
    //}

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    message_type make_device_buffer(std::size_t size, int id);
    //{
    //    return m_ctxt->template make_device_buffer<unsigned char>(size, id);
    //}
#endif
};

} // namespace ghex
