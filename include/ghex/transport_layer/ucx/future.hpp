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
#ifndef INCLUDED_GHEX_TL_UCX_FUTURE_HPP
#define INCLUDED_GHEX_TL_UCX_FUTURE_HPP

#include "./request.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace ucx {

                template<typename RandomAccessIterator>
                static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                    const auto count = last-first;
                    if (count == 0) return last;
                    for (auto it = first; it!=last; ++it)
                        if (it->test())
                            return it;
                    return last;
                }

                template<typename RandomAccessIterator, typename Func>
                static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                    Func&& get)
                {
                    const auto count = last-first;
                    if (count == 0) return last;
                    for (auto it = first; it!=last; ++it)
                        if (get(*it).test())
                            return it;
                    return last;
                }

                /** @brief future template for non-blocking communication */
                template<typename T, typename ThreadPrimitives>
                struct future_t
                {
                    using value_type  = T;
                    using handle_type = request_ft<ThreadPrimitives>;

                    value_type m_data;
                    handle_type m_handle;

                    future_t(value_type&& data, handle_type&& h) 
                    :   m_data(std::move(data))
                    ,   m_handle(std::move(h))
                    {}
                    future_t(const future_t&) = delete;
                    future_t(future_t&&) = default;
                    future_t& operator=(const future_t&) = delete;
                    future_t& operator=(future_t&&) = default;

                    void wait() noexcept
                    {
                        m_handle.wait();
                    }

                    bool test() noexcept
                    {
                        return m_handle.test();
                    }

                    bool ready() noexcept
                    {
                        return m_handle.test();
                    }

                    [[nodiscard]] value_type get()
                    {
                        wait(); 
                        return std::move(m_data); 
                    }

                    /** Cancel the future.
                      * @return True if the request was successfully canceled */
                    bool cancel()
                    {
                        return m_handle.cancel();
                    }

                    template<typename RandomAccessIterator>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                        return ::gridtools::ghex::tl::ucx::test_any(first,last);
                    }

                    template<typename RandomAccessIterator, typename Func>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                        Func&& get) {
                        return ::gridtools::ghex::tl::ucx::test_any(first,last,std::forward<Func>(get));
                    }
                };

                template<typename ThreadPrimitives>
                struct future_t<void, ThreadPrimitives>
                {
                    using handle_type = request_ft<ThreadPrimitives>;

                    handle_type m_handle;

                    future_t() noexcept = default; 
                    future_t(handle_type&& h) 
                    :   m_handle(std::move(h))
                    {}
                    future_t(const future_t&) = delete;
                    future_t(future_t&&) = default;
                    future_t& operator=(const future_t&) = delete;
                    future_t& operator=(future_t&&) = default;

                    void wait() noexcept
                    {
                        m_handle.wait();
                    }

                    bool test() noexcept
                    {
                        return m_handle.test();
                    }

                    bool ready() noexcept
                    {
                        return m_handle.test();
                    }

                    void get()
                    {
                        wait(); 
                    }

                    bool cancel()
                    {
                        return m_handle.cancel();
                    }

                    template<typename RandomAccessIterator>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                        return ::gridtools::ghex::tl::ucx::test_any(first,last);
                    }

                    template<typename RandomAccessIterator, typename Func>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                        Func&& get) {
                        return ::gridtools::ghex::tl::ucx::test_any(first,last,std::forward<Func>(get));
                    }
                };

            } // namespace ucx
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_UCX_FUTURE_HPP */

