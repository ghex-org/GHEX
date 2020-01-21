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
#ifndef INCLUDED_GHEX_THREADS_MUTEX_ATOMIC_HPP
#define INCLUDED_GHEX_THREADS_MUTEX_ATOMIC_HPP

#include <mutex>
#include <atomic>

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace mutex {
                namespace atomic {

                    class mutex
                    {
                    private: // members
                        std::atomic<bool> m_flag;
                        
                        int& level() noexcept
                        {
                            static thread_local int i = 0;
                            return i;
                        }

                    public:
                        mutex() noexcept : m_flag(0) {}
                        mutex(const mutex&) = delete;
                        mutex(mutex&&) = delete;

                        inline bool try_lock() noexcept
                        {
                            bool expected = false;
                            if (m_flag.compare_exchange_weak(expected, true, std::memory_order_relaxed))
                            {
                                ++level();
                                return true;
                            }
                            else
                                return false;
                        }

                        inline bool try_unlock() noexcept
                        {
                            bool expected = true;
                            return m_flag.compare_exchange_weak(expected, false, std::memory_order_relaxed);
                        }
                           
                        inline void lock() noexcept
                        {
                            if (level()==0)
                                while (!try_lock()) {}
                            else
                                ++level();
                        } 

                        inline void unlock() noexcept
                        {
                            --level();
                            if (level()==0)
                                while (!try_unlock()) {}
                        } 
                    };

                    using lock_guard = std::lock_guard<mutex>;

                } // namespace atomic
            } // namespace mutex
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_MUTEX_ATOMIC_HPP */

