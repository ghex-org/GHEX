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
#ifndef INCLUDED_GHEX_TL_CONTEXT_HPP
#define INCLUDED_GHEX_TL_CONTEXT_HPP

#include <mpi.h>
#include <memory>
#include "./config.hpp"
#include "./communicator.hpp"
#include "./mpi/setup.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename TransportTag, typename ThreadPrimitives>
            class transport_context;

            template<class TransportTag, class ThreadPrimitives>
            class context
            {
            public: // member types
                using tag                    = TransportTag;
                using transport_context_type = transport_context<tag,ThreadPrimitives>;
                using communicator_type      = typename transport_context_type::communicator_type;
                using thread_primitives_type = ThreadPrimitives;
                using thread_token           = typename thread_primitives_type::token;

            private:
                MPI_Comm m_mpi_comm;
                thread_primitives_type m_thread_primitives;
                transport_context_type m_transport_context;
                int m_rank;
                int m_size;

            public:
                template<typename...Args>
                context(int num_threads, MPI_Comm comm, Args&&... args)
                    : m_mpi_comm{comm}
                    , m_thread_primitives(num_threads)
                    , m_transport_context{m_thread_primitives, std::forward<Args>(args)...}
                    , m_rank{ [](MPI_Comm c){ int r; GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(c,&r)); return r; }(comm) }
                    , m_size{ [](MPI_Comm c){ int s; GHEX_CHECK_MPI_RESULT(MPI_Comm_size(c,&s)); return s; }(comm) }
                {}

                context(const context&) = delete;
                context(context&&) = delete;

            public:
                MPI_Comm mpi_comm() const noexcept { return m_mpi_comm; }
                int rank() const noexcept { return m_rank; }
                int size() const noexcept { return m_size; }

                // per-rank thread primitives
                // thread-safe
                thread_primitives_type& thread_primitives() noexcept
                {
                    return m_thread_primitives;
                }

                // per-rank communicator
                // only serial part of code
                mpi::setup_communicator get_setup_communicator()
                {
                    return mpi::setup_communicator(m_mpi_comm);
                }

                // per-rank communicator
                // only serial part of code
                communicator_type get_serial_communicator()
                {
                    return m_transport_context.get_serial_communicator();
                }

                // per-thread communicator
                // thread-safe
                communicator_type get_communicator(const thread_token& t)
                {
                    return m_transport_context.get_communicator(t);
                }

                // thread-safe
                thread_token get_token() noexcept
                {
                    return m_thread_primitives.get_token();
                }
            };

            template<class TransportTag, class ThreadPrimitives>
            struct context_factory;

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CONTEXT_HPP */
