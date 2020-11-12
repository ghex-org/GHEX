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

#include <vector>
#include <memory>
#include <atomic>
#include <mutex>
#include <cassert>
#include <mpi.h>

namespace gridtools {
    namespace ghex {
        namespace tl {

            /**
               The barrier object synchronize threads or ranks, or both. When synchronizing
               ranks, it also progress the transport_layer::communicator.

               This facility id provided as a debugging tool, or as a seldomly used operation.
               Halo-exchanges doe not need, in general, to call barriers.

               Note on the implementation:

               The implementation follows a two-counter approach:
               First: one (atomic) counter is increased to the number of threads participating.
               Second: one (atomic) counter is decreased from the numner of threads

               The global barrier performs the up-counting while the thread that reaches the
               final value also perform the rank-barrier. After that the downward count is
               performed as usual.

               This is why the barrier is split into is_node1 and in_node2. in_node1 returns
               true to the thread selected to run the rank_barrier in the full barrier.
             */
            struct barrier_t
            {

            private: // members
                size_t                   m_threads;
                mutable std::atomic<size_t> b_count{0};
                mutable std::atomic<size_t> b_count2;

            public: // ctors
                friend struct test_barrier;

                barrier_t(size_t n_threads = 1) : m_threads{n_threads}, b_count2{m_threads} {
                }

                barrier_t(const barrier_t&) = delete;
                barrier_t(barrier_t&&) = delete;

            public: // public member functions

                int size() const noexcept
                {
                    return m_threads;
                }

                /**
                 * This is the most general barrier, it synchronize threads and ranks.
                 *
                 * @param tlcomm The communicator object associated with the thread.
                 */
                template <typename TLCommunicator>
                void operator()(TLCommunicator& tlcomm) const
                {
                    if (in_node1(tlcomm))
                        rank_barrier(tlcomm);
                    else
                    {
                        while(b_count2 == m_threads)
                            tlcomm.progress();
                    }
                    in_node2(tlcomm);
                }

                /**
                 * This function can be used to synchronize ranks.
                 * Only one thread per rank must call this function.
                 * If other threads exist, they hace to be synchronized separately,
                 * maybe using the in_node function.
                 *
                 * @param tlcomm The communicator object associated with the thread.
                 */
                template <typename TLCommunicator>
                void rank_barrier(TLCommunicator& tlcomm) const
                {
                    MPI_Request req = MPI_REQUEST_NULL;
                    int flag;
                    MPI_Ibarrier(tlcomm.mpi_comm(), &req);
                    while(true) {
                        tlcomm.progress();
                        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                        if(flag) break;
                    }
                }

                /**
                 * This function synchronize the threads in a rank. The number of threads that need to participate
                 * is indicated in the construction of the barrier object, whose reference is shared among the
                 * participating threads.
                 *
                 * @param tlcomm The communicator object associated with the thread.
                 */
                template <typename TLCommunicator>
                void in_node(TLCommunicator& tlcomm) const
                {
                    in_node1(tlcomm);
                    in_node2(tlcomm);
                 }

            private:
                template <typename TLCommunicator>
                bool in_node1(TLCommunicator& tlcomm) const
                {
                    size_t expected = b_count;
                    while (!b_count.compare_exchange_weak(expected, expected+1, std::memory_order_relaxed))
                        expected = b_count;
                    if (expected == m_threads-1)
                        {
                            b_count.store(0);
                            return true;
                        } else {
                            while (b_count != 0) { tlcomm.progress(); }
                            return false;
                        }
                }

                template <typename TLCommunicator>
                void in_node2(TLCommunicator& tlcomm) const
                {
                    size_t ex = b_count2;
                    while(!b_count2.compare_exchange_weak(ex, ex-1, std::memory_order_relaxed))
                        ex = b_count2;
                    if (ex == 1) {
                        b_count2.store(m_threads);
                    } else {
                        while (b_count2 != m_threads) { tlcomm.progress(); }
                    }
                }

            };

        }
    }
}
