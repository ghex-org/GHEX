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
#ifndef INCLUDED_GHEX_TL_CONTEXT_HPP
#define INCLUDED_GHEX_TL_CONTEXT_HPP

#include <mpi.h>
#include <memory>
#include "./config.hpp"
#include "./tags.hpp"
#include "./mpi/setup.hpp"
#include "./mpi/rank_topology.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            namespace detail {

                static MPI_Comm clone_mpi_comm(MPI_Comm mpi_comm) {
                    // clone the communicator first to be independent of user calls to the mpi runtime
                    MPI_Comm new_comm;
                    MPI_Comm_dup(mpi_comm, &new_comm);
                    return new_comm;
                }

            }

            /** @brief Forward declaration of the context factory */
            template<class TransportTag>
            struct context_factory;

            /** @brief class providing access to the transport backend. Note, that this class can only be created using
              * the factory class `context_factory`.
              * @tparam TransportTag the transport tag (mpi_tag, ucx_tag, ...)
              */
            template<class TransportContext>
            class context
            {
            public: // member types
                using transport_context_type = TransportContext;
                using tag                    = typename transport_context_type::tag;
                using communicator_type      = typename transport_context_type::communicator_type;

                friend class context_factory<tag>;

            private: // members
                MPI_Comm m_mpi_comm;
                mpi::rank_topology m_rank_topology;
                transport_context_type m_transport_context;
                int m_rank;
                int m_size;

            private: // private ctor
                template<typename...Args>
                context(MPI_Comm comm, Args&&... args)
                    : m_mpi_comm{comm}
                    , m_rank_topology{comm}
                    , m_transport_context{m_rank_topology, std::forward<Args>(args)...}
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(comm) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(comm) }
                {}

            public: // ctors
                context(const context&) = delete;
                context(context&&) = delete;

                ~context()
                {
                    MPI_Comm_free(&m_mpi_comm);
                }


            public: // member functions
                MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }
                int rank() const noexcept { return m_rank; }
                int size() const noexcept { return m_size; }

                /** @brief return a per-rank communicator.
                  * This function is not thread-safe and should only be used in the serial part of the code. */
                communicator_type get_serial_communicator()
                {
                    return m_transport_context.get_serial_communicator();
                }

                /** @brief return a per-thread communicator.
                  * This function is thread-safe. */
                communicator_type get_communicator()
                {
                    return m_transport_context.get_communicator();
                }
            };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CONTEXT_HPP */
