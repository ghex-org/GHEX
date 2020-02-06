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
#ifndef INCLUDED_TL_MPI_CONTEXT_HPP
#define INCLUDED_TL_MPI_CONTEXT_HPP

#include "../context.hpp"
#include "./communicator.hpp"
#include "../communicator.hpp"

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename ThreadPrimitives>
            struct transport_context<mpi_tag, ThreadPrimitives>
            {
                using thread_primitives_type = ThreadPrimitives;
                using communicator_type = communicator<mpi::communicator<thread_primitives_type>>;
                using thread_token = typename thread_primitives_type::token;
                using shared_state_type = typename communicator_type::shared_state_type;
                using state_type = typename communicator_type::state_type;
                using state_ptr = std::unique_ptr<state_type>;
                using state_vector = std::vector<state_ptr>;

                thread_primitives_type& m_thread_primitives;
                MPI_Comm m_comm;
                std::vector<thread_token>  m_tokens;
                shared_state_type m_shared_state;
                state_type m_state;
                state_vector m_states;

                template<typename... Args>
                transport_context(ThreadPrimitives& tp, MPI_Comm mpi_comm, Args&&...)
                : m_thread_primitives(tp)
                , m_comm{mpi_comm}
                , m_tokens(tp.size())
                , m_shared_state(mpi_comm, this, &tp)
                , m_state(nullptr)
                , m_states(tp.size())
                {}

                communicator_type get_serial_communicator()
                {
                    return {&m_shared_state, &m_state};
                }

                communicator_type get_communicator(const thread_token& t)
                {
                    if (!m_states[t.id()])
                    {
                        m_tokens[t.id()] = t;
                        m_states[t.id()] = std::make_unique<state_type>(&m_tokens[t.id()]);
                    }
                    return {&m_shared_state, m_states[t.id()].get()};
                }

            };

            template<class ThreadPrimitives>
            struct context_factory<mpi_tag, ThreadPrimitives>
            {
                static std::unique_ptr<context<mpi_tag, ThreadPrimitives>> create(int num_threads, MPI_Comm mpi_comm)
                {
                    auto new_comm = detail::clone_mpi_comm(mpi_comm);
                    return std::unique_ptr<context<mpi_tag, ThreadPrimitives>>{
                        new context<mpi_tag,ThreadPrimitives>{num_threads, new_comm, new_comm}};
                }
            };

        }
    }
}

#endif /* INCLUDED_TL_MPI_CONTEXT_HPP */


