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
#ifndef INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP
#define INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP

#include <mutex>
#include <pthread.h>
#include <sched.h>

#ifdef _MACOS_GHEX_
// This code was cut&pasted from stackoverflow - copyright of the person posting it
typedef int pthread_spinlock_t;

int pthread_spin_init(pthread_spinlock_t *lock, int pshared) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}

int pthread_spin_destroy(pthread_spinlock_t *lock) {
    return 0;
}

int pthread_spin_lock(pthread_spinlock_t *lock) {
    while (1) {
        int i;
        for (i=0; i < 10000; i++) {
            if (__sync_bool_compare_and_swap(lock, 0, 1)) {
                return 0;
            }
        }
        sched_yield();
    }
}

int pthread_spin_trylock(pthread_spinlock_t *lock) {
    if (__sync_bool_compare_and_swap(lock, 0, 1)) {
        return 0;
    }
    return EBUSY;
}

int pthread_spin_unlock(pthread_spinlock_t *lock) {
    __asm__ __volatile__ ("" ::: "memory");
    *lock = 0;
    return 0;
}
#endif

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace mutex {
                namespace pthread_spin {

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
                            while (!try_lock()) { sched_yield(); }
                        }

                        inline void unlock() noexcept
                        {
                            pthread_spin_unlock(&m_lock);
                        }
                    };

                    using lock_guard = std::lock_guard<mutex>;

                } // namespace pthread_spin
            } // namespace mutex
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_MUTEX_PTHREAD_SPIN_HPP */
