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
#ifndef INCLUDED_GHEX_THREADS_OMP_PRIMITIVES_HPP
#define INCLUDED_GHEX_THREADS_OMP_PRIMITIVES_HPP

#include <omp.h>
#include <memory>
#include <boost/callable_traits.hpp>
#include "../mutex/pthread_spin/mutex.hpp"
#include "../mutex/atomic/mutex.hpp"

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace omp {
           
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
                        
                        friend primitives;

                        token_impl(id_type id) noexcept
                        : m_id(id) {}

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

                    using mutex_type = ::gridtools::ghex::threads::mutex::pthread_spin::mutex;
                    using lock_type  = ::gridtools::ghex::threads::mutex::pthread_spin::lock_guard;
                    //using mutex_type = ::gridtools::ghex::threads::mutex::atomic::mutex;
                    //using lock_type  = ::gridtools::ghex::threads::mutex::atomic::lock_guard;
                    //using mutex_type = std::mutex;
                    //using lock_type  = std::lock_guard<mutex_type>;

                private: // members
                    const int                m_num_threads;
                    std::vector<std::unique_ptr<token_impl>> m_tokens;
                public:
                    mutable mutex_type       m_mutex;

                public: // ctors
                    primitives(int num_threads) noexcept
                    : m_num_threads(num_threads)
                    , m_tokens(num_threads)
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
                        const int id = omp_get_thread_num();
                        m_tokens[id].reset( new token_impl{id} );
                        return {m_tokens[id].get()};
                    }

                    inline void barrier(token&) const
                    {
                        #pragma omp barrier
                    }

                    template <typename F>
                    inline void single(token&, F && f) const
                    {
                        #pragma omp single
                        f();
                    }

                    template <typename F>
                    inline void master(token&, F && f) const
                    {
                        #pragma omp master
                        f();
                    }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) const
                    {
                        //#pragma omp critical
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

            } // namespace omp
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_THREADS_OMP_PRIMITIVES_HPP */

