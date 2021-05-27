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
#ifndef INCLUDED_GHEX_TL_MPI_FUTURE_HPP
#define INCLUDED_GHEX_TL_MPI_FUTURE_HPP

#include <vector>
#include <iterator>
#include <algorithm>
#include "./request.hpp"

namespace gridtools{
    namespace ghex {
        namespace tl {
            namespace mpi {

                template<typename RandomAccessIterator>
                static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                    const auto count = last-first;
                    if (count == 0) return last;
                    // should we handle null requests ourselves?
                    //for (auto it = first; it!=last; ++it)
                    //    if (*it==MPI_REQUEST_NULL)
                    //        return it;
                    // maybe static needed to avoid unnecessary allocations
                    static thread_local std::vector<MPI_Request> reqs;
                    reqs.resize(0);
                    reqs.reserve(count);
                    //std::vector<MPI_Request> reqs;
                    reqs.reserve(count);
                    int indx, flag;
                    std::transform(first, last, std::back_inserter(reqs), [](auto& fut){
                        return fut.m_handle.m_req.m_struct; });
                    GHEX_CHECK_MPI_RESULT(
                        MPI_Testany(count, reqs.data(), &indx, &flag, MPI_STATUS_IGNORE));
                    if (flag && indx != MPI_UNDEFINED) return first+indx;
                    else return last;
                }

                template<typename RandomAccessIterator, typename Func>
                static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                    Func&& get)
                {
                    const auto count = last-first;
                    if (count == 0) return last;
                    // should we handle null requests ourselves?
                    //for (auto it = first; it!=last; ++it)
                    //    if (get(*it)==MPI_REQUEST_NULL)
                    //        return it;
                    // maybe static needed to avoid unnecessary allocations
                    static thread_local std::vector<MPI_Request> reqs;
                    reqs.resize(0);
                    reqs.reserve(count);
                    //std::vector<MPI_Request> reqs;
                    int indx, flag;
                    std::transform(first, last, std::back_inserter(reqs),
                        [&get](auto& x) { return get(x).m_handle.m_req.m_struct; });
                    GHEX_CHECK_MPI_RESULT(
                        MPI_Testany(count, reqs.data(), &indx, &flag, MPI_STATUS_IGNORE));
                    if (flag && indx != MPI_UNDEFINED) return first+indx;
                    else return last;
                }

                /** @brief future template for non-blocking communication */
                template<typename T>
                struct future_t
                {
                    using value_type  = T;
                    using handle_type = request_t;

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
                    
                    bool is_recv() const noexcept { return (m_handle.m_kind == request_kind::recv); }

                    /** Cancel the future.
                      * @return True if the request was successfully canceled */
                    bool cancel()
                    {
                        // we can  only cancel recv requests...
                        if  (is_recv()) {
                            GHEX_CHECK_MPI_RESULT(MPI_Cancel(&m_handle.get()));
                            MPI_Status st;
                            GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_handle.get(), &st));
                            int flag = false;
                            GHEX_CHECK_MPI_RESULT(MPI_Test_cancelled(&st, &flag));
                            return flag;
                        }
                        else
                            return false;
                    }

                    template<typename RandomAccessIterator>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                        return ::gridtools::ghex::tl::mpi::test_any(first,last);
                    }

                    template<typename RandomAccessIterator, typename Func>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                        Func&& get) {
                        return ::gridtools::ghex::tl::mpi::test_any(first,last,std::forward<Func>(get));
                    }
                };

                template<>
                struct future_t<void>
                {
                    using handle_type = request_t;

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

                    bool is_recv() const noexcept { return (m_handle.m_kind == request_kind::recv); }

                    bool cancel()
                    {
                        // we can  only cancel recv requests...
                        if  (is_recv()) {
                            GHEX_CHECK_MPI_RESULT(MPI_Cancel(&m_handle.get()));
                            MPI_Status st;
                            GHEX_CHECK_MPI_RESULT(MPI_Wait(&m_handle.get(), &st));
                            int flag = false;
                            GHEX_CHECK_MPI_RESULT(MPI_Test_cancelled(&st, &flag));
                            return flag;
                        }
                        else
                            return false;
                    }

                    template<typename RandomAccessIterator>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last) {
                        return ::gridtools::ghex::tl::mpi::test_any(first,last);
                    }

                    template<typename RandomAccessIterator, typename Func>
                    static RandomAccessIterator test_any(RandomAccessIterator first, RandomAccessIterator last,
                        Func&& get) {
                        return ::gridtools::ghex::tl::mpi::test_any(first,last,std::forward<Func>(get));
                    }
                };

            } // namespace mpi
        } // namespace tl
    } // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_TL_MPI_FUTURE_HPP */

