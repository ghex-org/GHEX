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
#ifndef INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP
#define INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP

#include <mutex>
#include <pthread.h>
#include <sched.h>


namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace mutex {
                namespace pthread_spin {

                    struct index {
                        int m_value = 0; 
                        int value() const noexcept { return m_value; }
                        int& value() noexcept { return m_value; }
                    };
                    static thread_local index m_index;
                    
                    class mutex
                    {
                    private: // members
                        pthread_spinlock_t m_lock;
                    public:
                        mutex() noexcept 
                        {
                            pthread_spin_init(&m_lock, PTHREAD_PROCESS_PRIVATE);
                        }
                        mutex(const mutex&) = delete;
                        mutex(mutex&&) = delete;
                        ~mutex()
                        {
                            pthread_spin_destroy(&m_lock);
                        }

                        inline bool try_lock() noexcept
                        {
                            return (pthread_spin_trylock(&m_lock)==0);
                        }
                           
                        inline void lock() noexcept
                        {
                            if (m_index.value())
                            {
                                ++m_index.value();
                                return;
                            }
                            while (!try_lock()) { sched_yield(); }
                            ++m_index.value();
                        } 

                        inline void unlock() noexcept
                        {
                            if (m_index.value()==1){
                                pthread_spin_unlock(&m_lock);
                            }
                            --m_index.value();
                        } 
                    };
                    
                    // thread_local mutex::index mutex::m_index;

                    using lock_guard = std::lock_guard<mutex>;

                } // namespace pthread_spin
            } // namespace mutex
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP */

