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

#include <memory>
#include <atomic>
#include <mutex>

namespace gridtools {
    namespace ghex {
        namespace tl {

            struct barrier_t
            {
            public: // member types
                using id_type = int;

                class token_impl
                {
                private: // members
                    id_type m_id;
                    int     m_epoch = 0;
                    bool    m_selected = false;

                    friend barrier_t;

                    token_impl(id_type id, int epoch) noexcept
                        : m_id(id), m_epoch(epoch), m_selected(id==0?true:false)
                    {}

                public: // ctors
                    token_impl(const token_impl&) = delete;
                    token_impl(token_impl&&) = default;

                public: // member functions
                    id_type id() const noexcept { return m_id; }
                };

                class token
                {
                private:
                    token_impl* impl = nullptr;
                    friend barrier_t;
                public:
                    token() = default;
                    token(token_impl* impl_) noexcept : impl{impl_} {}
                    token(const token&) = default;
                    token(token&&) = default;
                    token& operator=(const token&) = default;
                    token& operator=(token&&) = default;
                public:
                    id_type id() const noexcept { return impl->id();}
                };

            private: // members
                int                      m_ids{0};
                std::vector<std::unique_ptr<token_impl>> m_tokens;
                std::mutex               m_mutex;
                mutable volatile int     m_epoch{0};
                mutable std::atomic<int> b_count{0};

            public: // ctors
                barrier_t() = default;

                barrier_t(const barrier_t&) = delete;
                barrier_t(barrier_t&&) = delete;

            public: // public member functions

                int size() const noexcept
                {
                    return m_tokens.size();
                }

                inline token get_token() noexcept
                {
                    std::lock_guard<std::mutex> lock(m_mutex);
                    m_tokens.push_back( std::unique_ptr<token_impl>{new token_impl{m_ids,0}} );
                    m_ids++;
                    return {m_tokens.back().get()};
                }

                /**
                 * This is the most general barrier, it synchronize threads and ranks.
                 *
                 * @param t Token obtained by callinf get_token before.
                 * @param tlcomm The communicator object associated with the thread.
                 */
                template <typename TLCommunicator>
                void operator()(token& t, TLCommunicator& tlcomm) const
                {
                    int expected = b_count;
                    while (!b_count.compare_exchange_weak(expected, expected+1, std::memory_order_relaxed))
                        expected = b_count;
                    t.impl->m_epoch ^= 1;
                    t.impl->m_selected = (expected?false:true);
                    if (expected == m_tokens.size()-1)
                        {
                            MPI_Request req = MPI_REQUEST_NULL;
                            int flag;
                            MPI_Ibarrier(tlcomm.context().mpi_comm(), &req);
                            while(true) {
                                tlcomm.progress();
                                MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                                if(flag) break;
                            }
                            b_count.store(0);
                            m_epoch ^= 1;
                        }
                    while(t.impl->m_epoch != m_epoch) {tlcomm.progress();}
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
                    MPI_Ibarrier(tlcomm.context().mpi_comm(), &req);
                    while(true) {
                        tlcomm.progress();
                        MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
                        if(flag) break;
                    }
                }

                /**
                 * This function synchronize the threads in a rank. The threads must obtain a token
                 * before being able to call this function.
                 *
                 * @param t Token obtained by callinf get_token before.
                 * @param tlcomm The communicator object associated with the thread.
                 */
                template <typename TLCommunicator>
                void in_node(token& t, TLCommunicator& tlcomm) const
                {
                    int expected = b_count;
                    while (!b_count.compare_exchange_weak(expected, expected+1, std::memory_order_relaxed))
                        expected = b_count;
                    t.impl->m_epoch ^= 1;
                    t.impl->m_selected = (expected?false:true);
                    if (expected == m_tokens.size()-1)
                        {
                            tlcomm.progress();
                            b_count.store(0);
                            m_epoch ^= 1;
                        }
                    while(t.impl->m_epoch != m_epoch) {tlcomm.progress();}
                }

            };

        }
    }
}
