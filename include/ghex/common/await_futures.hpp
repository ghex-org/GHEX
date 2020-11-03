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
#ifndef INCLUDED_GHEX_COMMON_AWAIT_FUTURES_HPP
#define INCLUDED_GHEX_COMMON_AWAIT_FUTURES_HPP

#include <vector>
#include <numeric>
#include <algorithm>

namespace gridtools {
namespace ghex {

/** @brief wait for all futures in a range to finish and call 
  * a continuation with the future's value as argument. */
template<typename Future, typename Continuation>
inline void await_futures(std::vector<Future>& range, Continuation&& cont)
{
    static thread_local std::vector<int> index_list;
    index_list.resize(range.size());
    std::iota(index_list.begin(), index_list.end(), 0);
    while (index_list.size())
    {
        index_list.resize(
                std::remove_if(index_list.begin(), index_list.end(),
                    [&range, cont = std::forward<Continuation>(cont)](int idx)
                    {
                        if (range[idx].test())
                        {
                            cont(range[idx].get());
                            return true;
                        } else return false;
                    })
                - index_list.begin());
    }
}

/** @brief wait for all requests in a range to finish **/
template<typename Request>
inline void await_requests(std::vector<Request>& range)
{
    static thread_local std::vector<int> index_list;
    index_list.resize(range.size());
    std::iota(index_list.begin(), index_list.end(), 0);
    while (index_list.size())
    {
        index_list.resize(
                std::remove_if(index_list.begin(), index_list.end(),
                    [&range](int idx) { return range[idx].test(); })
                - index_list.begin());
    }
}

/** @brief wait for all requests in a range to finish and call 
  * a progress function regularly. */
template<typename Request, typename Progress>
inline void await_requests(std::vector<Request>& range, Progress&& progress)
{
    static thread_local std::vector<int> index_list;
    index_list.resize(range.size());
    std::iota(index_list.begin(), index_list.end(), 0);
    while (index_list.size())
    {
        progress();
        index_list.resize(
                std::remove_if(index_list.begin(), index_list.end(),
                    [&range](int idx) { return range[idx].test(); })
                - index_list.begin());
    }
}

} // namespace ghex
} // namespace gridtools

#endif // INCLUDED_GHEX_COMMON_AWAIT_FUTURES_HPP
