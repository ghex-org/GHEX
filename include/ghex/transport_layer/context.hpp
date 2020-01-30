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
#include "./tags.hpp"
#include "./mpi/setup.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            namespace detail {

                static MPI_Comm clone_mpi_comm(MPI_Comm mpi_comm) {
                    // clone the communicator first to be independent of user calls to the mpi runtime
                    MPI_Comm new_comm;
                    int topo_type;
                    GHEX_CHECK_MPI_RESULT(MPI_Topo_test(mpi_comm, &topo_type));
                    if (MPI_CART == topo_type)
                    {
                        // using a cart comm
                        int ndims;
                        GHEX_CHECK_MPI_RESULT(MPI_Cartdim_get(mpi_comm, &ndims));
                        auto dims = new int[ndims];
                        auto periods = new int[ndims];
                        auto coords = new int[ndims];
                        GHEX_CHECK_MPI_RESULT(MPI_Cart_get(mpi_comm, ndims, dims, periods, coords));
                        GHEX_CHECK_MPI_RESULT(MPI_Cart_create(mpi_comm, ndims, dims, periods, 0, &new_comm));
                        delete[] dims;
                        delete[] periods;
                        delete[] coords;
                    }
                    else
                    {
                        int rank;
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_rank(mpi_comm,&rank));
                        GHEX_CHECK_MPI_RESULT(MPI_Comm_split(mpi_comm, 0, rank, &new_comm));
                    }
                    return new_comm;
                }
                
            }

            /** @brief Forward declaration of the transport backend */
            template<typename TransportTag, typename ThreadPrimitives>
            class transport_context;

            /** @brief Forward declaration of the context factory */
            template<class TransportTag, class ThreadPrimitives>
            struct context_factory;

            /** @brief class providing access to the transport backend. Note, that this class can only be created using
              * the factory class `context_factory`. 
              * @tparam TransportTag the transport tag (mpi_tag, ucx_tag, ...)
              * @tparam ThreadPrimitives type for thread managment */
            template<class TransportTag, class ThreadPrimitives>
            class context
            {
            public: // member types
                using tag                    = TransportTag;
                using transport_context_type = transport_context<tag,ThreadPrimitives>;
                using communicator_type      = typename transport_context_type::communicator_type;
                using thread_primitives_type = ThreadPrimitives;
                using thread_token           = typename thread_primitives_type::token;

                friend class context_factory<TransportTag,ThreadPrimitives>;

            private: // members
                MPI_Comm m_mpi_comm;
                thread_primitives_type m_thread_primitives;
                transport_context_type m_transport_context;
                int m_rank;
                int m_size;

            private: // private ctor
                template<typename...Args>
                context(int num_threads, MPI_Comm comm, Args&&... args)
                    : m_mpi_comm{comm}
                    , m_thread_primitives(num_threads)
                    , m_transport_context{m_thread_primitives, std::forward<Args>(args)...}
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

                /** @brief return a reference to the thread-shared thread primitives instance.
                  * This function is thread-safe. */
                thread_primitives_type& thread_primitives() noexcept
                {
                    return m_thread_primitives;
                }

                /** @brief return a special per-rank setup communicator.
                  * This function is not thread-safe and should only be used in the serial part of the code. */
                mpi::setup_communicator get_setup_communicator()
                {
                    return mpi::setup_communicator(m_mpi_comm);
                }

                /** @brief return a per-rank communicator.
                  * This function is not thread-safe and should only be used in the serial part of the code. */
                communicator_type get_serial_communicator()
                {
                    return m_transport_context.get_serial_communicator();
                }

                /** @brief return a per-thread communicator.
                  * This function is thread-safe. */
                communicator_type get_communicator(const thread_token& t)
                {
                    return m_transport_context.get_communicator(t);
                }

                /** @brief return a per-thread thread token.
                  * This function is thread-safe. */
                thread_token get_token() noexcept
                {
                    return m_thread_primitives.get_token();
                }
            };

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CONTEXT_HPP */
