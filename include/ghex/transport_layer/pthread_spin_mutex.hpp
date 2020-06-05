#ifndef GHEX_PTHREAD_SPIN_MUTEX_HPP
#define GHEX_PTHREAD_SPIN_MUTEX_HPP

#include <pthread.h>

namespace gridtools {
    namespace ghex {
        namespace tl {
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
        }
    }
}
#endif
