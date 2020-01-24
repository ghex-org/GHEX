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

namespace gridtools {
    namespace ghex {
        namespace tl {

            template<typename ThreadPrimitives>
            struct transport_context<mpi_tag, ThreadPrimitives>
            {
                using thread_primitives_type = ThreadPrimitives;
                using communicator_type = mpi::communicator<thread_primitives_type>;
                //using parallel_context_type = parallel_context<ThreadPrimitives>;
                using thread_token = typename thread_primitives_type::token;

                thread_primitives_type& m_thread_primitives;
                MPI_Comm m_comm;

                template<typename... Args>
                transport_context(ThreadPrimitives& tp, MPI_Comm mpi_comm, Args&&...)
                : m_thread_primitives(tp)
                , m_comm{mpi_comm}
                {}

                communicator_type get_serial_communicator()
                {
                    return {m_comm,this};
                }

                communicator_type get_communicator(const thread_token& t)
                {
                    return {m_comm,this, t.id()};
                }

            };

            template<class ThreadPrimitives>
            struct context_factory<mpi_tag, ThreadPrimitives>
            {
                static std::unique_ptr<context<mpi_tag, ThreadPrimitives>> create(int num_threads, MPI_Comm mpi_comm)
                {
                    return std::make_unique<context<mpi_tag,ThreadPrimitives>>(num_threads, mpi_comm, mpi_comm);
                }
            };

        }
    }
}

#endif /* INCLUDED_TL_MPI_CONTEXT_HPP */


