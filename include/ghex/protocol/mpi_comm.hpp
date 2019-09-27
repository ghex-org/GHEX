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
#ifndef INCLUDED_MPI_COMM_HPP
#define INCLUDED_MPI_COMM_HPP

#include <mpi.h>
#include <string>
#include <stdexcept>

/*
// adapted from boost
#define GHEX_MPI_CHECK_RESULT( MPIFunc, Args )                                  \
{                                                                               \
   int _check_result = MPIFunc Args;                                            \
   assert(_check_result == MPI_SUCCESS);                                        \
   if (_check_result != MPI_SUCCESS)                                            \
        throw std::runtime_error(#MPIFunc);                                     \
}
*/
//#ifdef NDEBUG
//    #define GHEX_CHECK_MPI_RESULT(x) x;
//#else
    #define GHEX_CHECK_MPI_RESULT(x) \
    if (x != MPI_SUCCESS)           \
        throw std::runtime_error("GHEX Error: MPI Call failed " + std::string(#x) + " in " + std::string(__FILE__) + ":" + std::to_string(__LINE__));
//#endif


namespace gridtools {
    namespace ghex {
        namespace mpi {

            /** @brief thin wrapper around MPI_Request */
            struct request
            {
                MPI_Request m_req;
                void wait()
                {
                    MPI_Status status;
                    GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_req, &status));
                }
                bool test()
                {
                    MPI_Status result;
                    int flag = 0;
                    GHEX_CHECK_MPI_RESULT(MPI_Test(&m_req, &flag, &result));
                    //return flag != 0? optional<status>(result) : optional<status>();
                    return flag != 0;
                }
            };

            /** @brief thin wrapper around MPI_Status */
            struct status
            {
                MPI_Status m_status;
                int source() const { return m_status.MPI_SOURCE; }
                int tag() const { return m_status.MPI_TAG; }
                int error() const { return m_status.MPI_ERROR; }
                bool cancelled() const
                {
                    int flag = 0;
                    GHEX_CHECK_MPI_RESULT(MPI_Test_cancelled(&m_status, &flag));
                    return flag != 0;
                }
                operator       MPI_Status&()       { return m_status; }
                operator const MPI_Status&() const { return m_status; }
            };

            struct comm_take_ownership_tag{};
            static constexpr comm_take_ownership_tag comm_take_ownership;

            /** @brief thin wrapper around MPI_Comm */
            struct mpi_comm
            {
                MPI_Comm m_comm;
                int m_rank;
                int m_size;
                bool m_owns = false;

                operator MPI_Comm() const { return m_comm; }

                mpi_comm() noexcept
                : mpi_comm(MPI_COMM_WORLD)
                {}

                mpi_comm(MPI_Comm c) noexcept
                : m_comm{c}
                , m_rank{ [&c](){ int r; MPI_Comm_rank(c,&r); return r; }() }
                , m_size{ [&c](){ int s; MPI_Comm_size(c,&s); return s; }() }
                {}

                mpi_comm(MPI_Comm c, comm_take_ownership_tag) noexcept
                : mpi_comm(c)
                { m_owns = true; }

                mpi_comm(const mpi_comm&) = delete;
                mpi_comm(mpi_comm&&) = default;

                ~mpi_comm()
                {
                    if (m_owns && m_comm != MPI_COMM_WORLD)
                        MPI_Comm_free(&m_comm);
                }

                inline int rank() const noexcept { return m_rank; }
                inline int size() const noexcept { return m_size; }
                void barrier() const { MPI_Barrier(m_comm); }
            };

        } // namespace mpi
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_MPI_COMM_HPP */

