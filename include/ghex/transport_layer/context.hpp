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
#include "../threads/context.hpp"
//#include "../common/moved_bit.hpp"
#include "./config.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename TransportTag>
            class transport_context;
            
            template<typename TransportTag>
            class context;

            class parallel_context;

            class mpi_world
            {
            private:
                MPI_Comm  m_comm;
                int       m_rank;
                int       m_size;
                bool      m_owning = false;

                mpi_world(int& argc, char**& argv)
                {
#if defined(GHEX_THREAD_SINGLE)
                    MPI_Init(&argc, &argv);
#elif defined(GHEX_MPI_USE_GHEX_LOCKS)
                    int provided;
                    MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &provided);
#else
                    int provided;
                    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
#endif
                    m_comm = MPI_COMM_WORLD;
                    MPI_Comm_rank(m_comm, &m_rank);
                    MPI_Comm_size(m_comm, &m_size);
                    m_owning = true;
                }

                mpi_world(MPI_Comm comm)
                : m_comm(comm)
                , m_rank{ [comm]() { int r; MPI_Comm_rank(comm, &r); return r; }() }
                , m_size{ [comm]() { int r; MPI_Comm_size(comm, &r); return r; }() }
                {}

                mpi_world(const mpi_world&) = delete;
                mpi_world(mpi_world&&) = delete;

                friend class parallel_context;

            public:
                ~mpi_world()
                {
                    if (m_owning) MPI_Finalize();
                }

            public:
                inline int rank() const noexcept { return m_rank; }
                inline int size() const noexcept { return m_size; }
                operator MPI_Comm() const noexcept { return m_comm; }
            };

            class parallel_context
            {
            public: // members
                using thread_context_type = ::gridtools::ghex::threads::context;
                using thread_token        = typename thread_context_type::token;

            private:
                mpi_world           m_world;
            public:
                thread_context_type m_thread_context;

            public:
                template<typename... Args>
                parallel_context(int num_threads, int& argc, char**& argv, Args&&...) noexcept
                : m_world(argc,argv)
                , m_thread_context(num_threads)
                {}

                template<typename... Args>
                parallel_context(int num_threads, MPI_Comm comm, Args&&...) noexcept
                : m_world(comm)
                , m_thread_context(num_threads)
                {}

                parallel_context(const parallel_context&) = delete;
                parallel_context(parallel_context&&) = delete;

                //template<typename Transport>
                //friend class context<Transport>;

            public:
                // thread-safe
                const mpi_world& world() const { return m_world; }
                // thread-safe
                const thread_context_type& thread_context() const { return m_thread_context; }
                // thread-safe
                void barrier(thread_token& t) const
                {
                    m_thread_context.barrier(t);
                    m_thread_context.single(t, [this]() { MPI_Barrier(m_world.m_comm); } );
                    m_thread_context.barrier(t);
                }
            };

            template<class TransportTag>
            class context
            {
            public: // member types
                using tag                    = TransportTag;
                using transport_context_type = transport_context<tag>;
                using communicator_type      = typename transport_context_type::communicator_type;
                using thread_token           = parallel_context::thread_token;

            private:
                std::unique_ptr<parallel_context> m_pc;
                transport_context_type            m_tc;

            public:
                template<typename...Args>
                context(int num_threads, Args&&... args)
                : m_pc{std::make_unique<parallel_context>(num_threads, std::forward<Args>(args)...)}
                , m_tc{*m_pc, std::forward<Args>(args)...}
                {}

                /*template<typename...Args>
                context(int& argc, char**& argv, MPI_Comm comm, int num_threads=1, M, Args&&... args)
                : m_pc{std::make_unique<parallel_context>(comm, num_threads)}
                , m_tc{*m_pc, argc, argv, std::forward<Args>(args)...}
                {}*/

            public:
                // thread-safe
                communicator_type get_communicator(const thread_token& t)
                {
                    return m_pc->m_thread_context.critical(
                        [this,&t]() mutable { return m_tc.get_communicator(t.id()); }
                    );
                }

                // thread-safe
                thread_token get_token() noexcept
                {
                    return m_pc->m_thread_context.get_token();
                }
            };    

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CONTEXT_HPP */

