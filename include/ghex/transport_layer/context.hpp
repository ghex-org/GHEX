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
                thread_primitives_type m_thread_primitives;
                transport_context_type m_transport_context;

            public:
                template<typename...Args>
                context(int num_threads, Args&&... args)
                    : m_thread_primitives(num_threads)
                    , m_transport_context{m_thread_primitives, std::forward<Args>(args)...}
                {}

                context(const context&) = delete;
                context(context&&) = delete;

            public:

                /*const */thread_primitives_type& thread_primitives() /*const*/ noexcept
                {
                    return m_thread_primitives;
                }

                mpi::setup_communicator get_setup_communicator()
                {
                    // TODO: what about this?
                    // return mpi::setup_communicator(m_parallel_context.world());
                }

                // per-rank communicator (recv)
                // thread-safe
                communicator_type get_serial_communicator()
                {
                    return m_transport_context.get_serial_communicator();
                }

                // per-thread communicator (send)
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

                // thread-safe
                void barrier(thread_token& t) /*const*/
                {
                    m_thread_primitives.barrier(t);
                    m_transport_context.barrier();
                    m_thread_primitives.barrier(t);
                }

            };

            template<class TransportTag, class ThreadPrimitives>
            struct context_factory
            {
                template<typename DB>
                static std::unique_ptr<context<TransportTag, ThreadPrimitives>> create(int num_threads, DB&& db)
                {
                    return std::make_unique<context<TransportTag,ThreadPrimitives>>(num_threads, std::forward<DB>(db));
                }
            };
            //{
            //    context<Transport_Tag,ThreadPrimitives>* create(num_threads, MPI_Comm);
            //};

        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_CONTEXT_HPP */
