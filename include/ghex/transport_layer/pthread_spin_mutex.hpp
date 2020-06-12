#ifndef GHEX_PTHREAD_SPIN_MUTEX_HPP
#define GHEX_PTHREAD_SPIN_MUTEX_HPP

#include <pthread.h>

namespace gridtools {
    namespace ghex {
        namespace tl {
            namespace pthread_spin {

                class recursive_mutex
                {
                private: // members
                    pthread_spinlock_t m_lock;

                    int& level() noexcept
                    {
                        static thread_local int i = 0;
                        return i;
                    }

                public:
                    recursive_mutex() noexcept
                    {
                        pthread_spin_init(&m_lock, PTHREAD_PROCESS_PRIVATE);
                    }
                    recursive_mutex(const recursive_mutex&) = delete;
                    recursive_mutex(recursive_mutex&&) = delete;
                    ~recursive_mutex()
                    {
                        pthread_spin_destroy(&m_lock);
                    }

                    inline bool try_lock() noexcept
                    {
                        if (pthread_spin_trylock(&m_lock)==0)
                            {
                                ++level();
                                return true;
                            }
                        else
                            return false;
                    }

                    inline void lock() noexcept
                    {
                        if (level()==0)
                            while (!try_lock()) { sched_yield(); }
                        else
                            ++level();
                    }

                    inline void unlock() noexcept
                    {
                        --level();
                        if (level()==0)
                            pthread_spin_unlock(&m_lock);
                    }
                };

                using lock_guard = std::lock_guard<recursive_mutex>;

            } // namespace pthread_spin
        }
    }
}
#endif
