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
#ifndef INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP
#define INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP

#include <atomic>
#include <boost/callable_traits.hpp>
#include "./mutex.hpp"

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace atomic {
           
                template<typename F>
                using void_return_type = typename std::enable_if<
                    std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    void>::type;

                template<typename F>
                using return_type = typename std::enable_if<
                    !std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    boost::callable_traits::return_type_t<F>>::type;

#ifndef GHEX_THREAD_SINGLE
                struct primitives
                {
                public: // member types
                    using id_type = int;

                    class token
                    {
                    private: // members
                        id_type m_id;
                        int     m_epoch = 0;
                        bool    m_selected = false;
                        
                        friend primitives;

                        token(id_type id, int epoch) noexcept
                        : m_id(id), m_epoch(epoch), m_selected(id==0?true:false)
                        {}

                    public: // ctors
                        token(const token&) = delete;
                        token(token&&) = default;

                    public: // member functions
                        id_type id() const noexcept { return m_id; }
                    };

                    using mutex_type = atomic_mutex;
                    using lock_type  = lock_guard<mutex_type>;

                private: // members
                    const int                m_num_threads;
                    std::atomic<int>         m_ids;
                    mutable volatile int     m_epoch;
                    mutable std::atomic<int> b_count;
                    mutable mutex_type       m_mutex;

                public: // ctors
                    primitives(int num_threads) noexcept
                    : m_num_threads(num_threads)
                    , m_ids(0)
                    , m_epoch(0)
                    , b_count(0)
                    {} 

                    primitives(const primitives&) = delete;
                    primitives(primitives&&) = delete;

                public: // public member functions
                    inline token get_token() noexcept
                    {
                        return {(int)m_ids++,0};
                    }

                    inline void barrier(token& t) const noexcept
                    {
                        int expected = b_count; 
                        while (!b_count.compare_exchange_weak(expected, expected+1, std::memory_order_relaxed))
                            expected = b_count;
                        t.m_epoch ^= 1;
                        t.m_selected = (expected?false:true);
                        if (expected == m_num_threads-1)
                        {
                            b_count.store(0);
                            m_epoch ^= 1;
                        }
                        while(t.m_epoch != m_epoch) {}
                    }

                    template <typename F>
                    inline void single(token& t, F && f) const noexcept
                    {
                        if (t.m_selected) {
                            f();
                        }
                    }

                    template <typename F>
                    inline void master(token& t, F && f) const noexcept
                    {
                        if (t.m_id == 0) {
                            f();
                        }
                    }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) const noexcept
                    {
                        lock_type l(m_mutex);
                        f();
                    }

                    template <typename F>
                    inline return_type<F> critical(F && f) const noexcept
                    {
                        lock_type l(m_mutex);
                        return f();
                    }
                };
#else
                struct primitives
                {
                public: // member types
                    using id_type = int;

                    class token
                    {
                    private: // members
                        id_type m_id;
                        
                        friend primitives;

                        token(id_type id) noexcept
                        : m_id(id)
                        {}

                    public: // ctors
                        token(const token&) = delete;
                        token(token&&) = default;

                    public: // member functions
                        id_type id() const noexcept { return m_id; }
                    };

                    using mutex_type = atomic_mutex;
                    using lock_type  = lock_guard<mutex_type>;

                private: // members

                public: // ctors
                    primitives(int=1) noexcept
                    {} 

                    primitives(const primitives&) = delete;
                    primitives(primitives&&) = delete;

                public: // public member functions
                    inline token get_token() noexcept { return {0}; }

                    inline void barrier(token& t) noexcept {}

                    template <typename F>
                    inline void single(token& t, F && f) const noexcept { f(); }

                    template <typename F>
                    inline void master(token& t, F && f) const noexcept { f(); }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) const noexcept { f(); }

                    template <typename F>
                    inline return_type<F> critical(F && f) const noexcept { return f(); }
                };
#endif
            } // namespace atomic
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP */

