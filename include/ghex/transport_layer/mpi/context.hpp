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

            template<>
            struct transport_context<mpi_tag>
            {
                using communicator_type = communicator<mpi_tag>;

                parallel_context& m_pc;

                template<typename... Args>
                transport_context(parallel_context& pc, Args&&...)
                : m_pc(pc)
                {}

                communicator_type get_communicator(int) const
                {
                    return {(MPI_Comm)(m_pc.world())};
                }

            };

        }
    }
}

#endif /* INCLUDED_TL_MPI_CONTEXT_HPP */


