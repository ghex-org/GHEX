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
#ifndef INCLUDED_GHEX_THREADS_STD_PRIMITIVES_HPP
#define INCLUDED_GHEX_THREADS_STD_PRIMITIVES_HPP

#include <thread>
#include <condition_variable>
#include <atomic>
#include <cassert>
#include <vector>
#include <boost/callable_traits.hpp>
#include <mutex>

namespace gridtools {
    namespace ghex {
        namespace threads {
            namespace std_thread {
                
                template<typename F>
                using void_return_type = typename std::enable_if<
                    std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    void>::type;

                template<typename F>
                using return_type = typename std::enable_if<
                    !std::is_same<boost::callable_traits::return_type_t<F>,void>::value, 
                    boost::callable_traits::return_type_t<F>>::type;

                struct primitives {
                    using id_type = int;
                private:

                    class token_impl
                    {
                    private: // members
                        int m_epoch = 0;
                        id_type m_id;
                        bool m_selected = false;

                        friend primitives;

                        token_impl(id_type id) noexcept
                        : m_id(id) {}

                    public: // ctors
                        token_impl(const token_impl&) = delete;
                        token_impl(token_impl&&) = default;

                    public: // member functions
                        id_type id() const noexcept { return m_id; }
                        int epoch() const noexcept { return m_epoch; }
                        void flip_epoch() noexcept { m_epoch ^= 1; }
                        void set_selected(bool v) noexcept { m_selected = v; }
                        bool is_selected() const noexcept { return m_selected; }
                    };

                public:
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
                    protected:
                        int epoch() const noexcept { return impl->epoch();}
                        void flip_epoch() noexcept { impl->flip_epoch(); }
                        void set_selected(bool v) noexcept { impl->set_selected(v); }
                        bool is_selected() const noexcept { return impl->is_selected(); }
                    };
                private:
                    std::atomic<int> m_id_counter;
                    const int m_num_threads;
                    std::mutex m_guard;
                    std::mutex m_cv_guard;
                    int m_barrier_cnt[2];
                    int m_up_counter[2];
                    std::vector<std::condition_variable> m_cv_down, m_cv_up;
                    std::vector<std::unique_ptr<token_impl>> m_tokens;

                public:
                    primitives(int n)
                        : m_id_counter{0}
                        , m_num_threads{n}
                        , m_barrier_cnt{m_num_threads, m_num_threads}
                        , m_up_counter{0,0}
                        , m_cv_down(2)
                        , m_cv_up(2)
                        , m_tokens(n)
                    { }

                    primitives(const primitives&) = delete;
                    primitives(primitives&&) = delete;


                    int size() const noexcept {
                        return m_num_threads;
                    }

                    token get_token() noexcept {
                        int new_id = m_id_counter.fetch_add(1);
                        assert(new_id < m_num_threads);
                        m_tokens[new_id].reset( new token_impl{new_id} );
                        if (new_id == 0) {
                            m_tokens[new_id]->set_selected(true);
                        }
                        return {m_tokens[new_id].get()};
                    }

                    void barrier(token& bt) /*const*/ {
                        if (m_num_threads < 2 )
                            return;
                        std::unique_lock<std::mutex> lock(m_cv_guard);

                        m_barrier_cnt[bt.epoch()]--;

                        if (m_barrier_cnt[bt.epoch()] == 0) {

                            m_cv_down[bt.epoch()].notify_all();
                            m_cv_up[bt.epoch()].wait(lock, [this, &bt] { return m_barrier_cnt[bt.epoch()] == m_num_threads;} );
                            bt.set_selected(false);
                        } else {
                            m_cv_down[bt.epoch()].wait(lock, [this, &bt] { return m_barrier_cnt[bt.epoch()] == 0; });

                            m_up_counter[bt.epoch()]++;

                            if (m_up_counter[bt.epoch()] == m_num_threads-1) {
                                m_up_counter[bt.epoch()] = 0;
                                m_barrier_cnt[bt.epoch()] = m_num_threads; // done by multiple threads, but this resets the counter
                                m_cv_up[bt.epoch()].notify_all();
                                bt.set_selected(true);
                            } else {
                                m_cv_up[bt.epoch()].wait(lock, [this, &bt] { return m_barrier_cnt[bt.epoch()] == m_num_threads;} );
                                bt.set_selected(false);
                            }
                        }
                        bt.flip_epoch();
                    }

                    template <typename F>
                    inline void_return_type<F> critical(F && f) //const
                    {
                        if (m_num_threads > 1 ) {
                            std::lock_guard<std::mutex> lock(m_guard);
                            f();
                        }
                        else
                            f();
                    }
                    template <typename F>
                    inline return_type<F> critical(F && f) //const
                    {
                        if (m_num_threads > 1 ) {
                            std::lock_guard<std::mutex> lock(m_guard);
                            return f();
                        }
                        else
                            return f();
                    }

                    template <typename F>
                    void master(token& bt, F && f) const { // Also this one should not be needed
                        if (bt.id() == 0) {
                            f();
                        }
                    }

                    template <typename F>
                    void single(token& bt, F && f) const { // Also this one should not be needed
                        if (bt.is_selected()) {
                            f();
                        }
                    }
                }; // struct threads
            } // namespace std_threads
        } // namespace threads
    } // namespace ghex
} // namespace gridtools

#endif

