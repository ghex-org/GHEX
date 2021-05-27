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
#ifndef INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_HPP
#define INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_HPP

#include <mutex>
#include <pthread.h>
#include <sched.h>

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace mutex {
                namespace pthread {

                    class mutex
                    {
                    private: // members
                        pthread_mutex_t m_mutex;
                    public:
                        mutex() noexcept 
                        {
                            pthread_mutexattr_t attr;
                            pthread_mutexattr_init(&attr);
                            pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);
                            pthread_mutex_init(&m_mutex, &attr);
                        }
                        mutex(const mutex&) = delete;
                        mutex(mutex&&) = delete;
                        ~mutex()
                        {
                            pthread_mutex_destroy(&m_mutex);
                        }

                        inline bool try_lock() noexcept
                        {
                            return (pthread_mutex_trylock(&m_mutex) == 0);
                        }
                           
                        inline void lock() noexcept
                        {
                            while (!try_lock()) { sched_yield(); }
                        } 

                        inline void unlock() noexcept
                        {
                            pthread_mutex_unlock(&m_mutex);
                        } 
                    };

                    using lock_guard = std::lock_guard<mutex>;

                } // namespace pthread
            } // namespace mutex
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_HPP */


