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
#ifndef INCLUDED_GHEX_TL_MPI_COMMUNICATOR_BASE_HPP
#define INCLUDED_GHEX_TL_MPI_COMMUNICATOR_BASE_HPP

#include "./error.hpp"
#include <memory>

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

                struct comm_take_ownership_tag{};
                static constexpr comm_take_ownership_tag comm_take_ownership;

                /** @brief thin wrapper around MPI_Comm */
                struct communicator_base
                {
                    using rank_type = int;
                    using size_type = int;
                    using tag_type  = int;
                    using comm_type = std::shared_ptr<MPI_Comm>;

                    comm_type m_comm;
                    rank_type m_rank;
                    size_type m_size;

                    communicator_base()
                    : m_comm{ new MPI_Comm{MPI_COMM_WORLD} }
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(MPI_COMM_WORLD) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(MPI_COMM_WORLD) }
                    {}

                    communicator_base(MPI_Comm c)
                    : m_comm{ new MPI_Comm{c} }
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(c) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(c) }
                    {}

                    communicator_base(MPI_Comm c, comm_take_ownership_tag)
                    : m_comm{ new MPI_Comm{c}, [](MPI_Comm* ptr) { MPI_Comm_free(ptr); } }
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(c) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(c) }
                    {}

                    communicator_base(const communicator_base&) = default;
                    communicator_base& operator=(const communicator_base&) = default;
                    communicator_base(communicator_base&&) noexcept = default;
                    communicator_base& operator=(communicator_base&&) noexcept = default;

                    /** @return rank of this process */
                    inline rank_type rank() const noexcept { return m_rank; }
                    /** @return size of communicator group*/
                    inline size_type size() const noexcept { return m_size; }

                    static void initialize() {}
		      static void finalize() {}

                    void barrier()
                    {
                        MPI_Barrier(*m_comm);
                    }

                    operator       MPI_Comm&()       noexcept { return *m_comm; }
                    operator const MPI_Comm&() const noexcept { return *m_comm; }
                          MPI_Comm& get()       noexcept { return *m_comm; }
                    const MPI_Comm& get() const noexcept { return *m_comm; }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_COMMUNICATOR_BASE_HPP */
