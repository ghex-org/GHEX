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
#ifndef INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP
#define INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP

#include <atomic>
#include <memory>
#include <boost/callable_traits.hpp>
#include "../mutex/atomic/mutex.hpp"

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

                struct primitives
                {
                public: // member types
                    using id_type = int;

                    class token_impl
                    {
                    private: // members
                        id_type m_id;
                        int     m_epoch = 0;
                        bool    m_selected = false;
                        
                        friend primitives;

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
                        friend primitives;
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

                    using mutex_type = ::gridtools::ghex::threads::mutex::atomic::mutex;
                    using lock_type  = ::gridtools::ghex::threads::mutex::atomic::lock_guard;

                private: // members
                    const int                m_num_threads;
                    std::vector<std::unique_ptr<token_impl>> m_tokens;
                    std::atomic<int>         m_ids;
                    mutable volatile int     m_epoch;
                    mutable std::atomic<int> b_count;
                    mutable mutex_type       m_mutex;

                public: // ctors
                    primitives(int num_threads) noexcept
                    : m_num_threads(num_threads)
                    , m_tokens(num_threads)
                    , m_ids(0)
                    , m_epoch(0)
                    , b_count(0)
                    {} 

                    primitives(const primitives&) = delete;
                    primitives(primitives&&) = delete;

                public: // public member functions
                    
                    int size() const noexcept
                    {
                        return m_num_threads;
                    }

                    inline token get_token() noexcept
                    {
                        const int id = m_ids++;
                        m_tokens[id].reset( new token_impl{id,0} );
                        return {m_tokens[id].get()};
                    }

                    inline void barrier(token& t) const
                    {
                        int expected = b_count; 
                        while (!b_count.compare_exchange_weak(expected, expected+1, std::memory_order_relaxed))
                            expected = b_count;
                        t.impl->m_epoch ^= 1;
                        t.impl->m_selected = (expected?false:true);
                        if (expected == m_num_threads-1)
                        {
                            b_count.store(0);
                            m_epoch ^= 1;
                        }
                        while(t.impl->m_epoch != m_epoch) {}
                    }

                    template <typename F>
                    inline void single(token& t, F && f) const
                    {
                        if (t.impl->m_selected) {
                            f();
                        }
                    }

                    template <typename F>
                    inline void master(token& t, F && f) const
                    {
                        if (t.impl->m_id == 0) {
                            f();
                        }
                    }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) const
                    {
                        lock_type l(m_mutex);
                        f();
                    }

                    template <typename F>
                    inline return_type<F> critical(F && f) const
                    {
                        lock_type l(m_mutex);
                        return f();
                    }
                };

            } // namespace atomic
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_ATOMIC_PRIMITIVES_HPP */

