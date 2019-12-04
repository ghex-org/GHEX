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
#ifndef INCLUDED_GHEX_THREADS_MUTEX_HPP
#define INCLUDED_GHEX_THREADS_MUTEX_HPP

#include <mutex>
#include <atomic>

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace atomic {

                class atomic_mutex
                {
                private: // members
                    std::atomic<bool> m_flag;
                public:
                    atomic_mutex() noexcept : m_flag(0) {}
                    atomic_mutex(const atomic_mutex&) = delete;
                    atomic_mutex(atomic_mutex&&) = delete;

                    inline bool try_lock() noexcept
                    {
                        bool expected = false;
                        return m_flag.compare_exchange_weak(expected, true, std::memory_order_relaxed);
                    }

                    inline bool try_unlock() noexcept
                    {
                        bool expected = true;
                        return m_flag.compare_exchange_weak(expected, false, std::memory_order_relaxed);
                    }
                       
                    inline void lock() noexcept
                    {
                        while (!try_lock()) {}
                    } 

                    inline void unlock() noexcept
                    {
                        while (!try_unlock()) {}
                    } 
                };

                template<typename Mutex>
                using lock_guard = std::lock_guard<Mutex>;
            } // namespace atomic
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_MUTEX_HPP */

