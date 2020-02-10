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

namespace gridtools {

    namespace ghex {
        
        /** @brief wait for all futures in a range to finish and call 
          * a continuation with the future's value as argument. */
        template<typename FutureRange, typename Continuation>
        void await_futures(FutureRange& range, Continuation&& cont)
        {
            int size = range.size();
            // make an index list (iota)
            std::vector<int> index_list(size);
            for (int i = 0; i < size; ++i)
                index_list[i] = i;
            // loop until all futures are ready
            while(size>0)
            {
                for (int j = 0; j < size; ++j)
                {
                    const auto k = index_list[j];
                    if (range[k].test())
                    {
                        if (j < --size)
                            index_list[j--] = index_list[size];
                        cont(range[k].get());
                    }
                }
            }
        }

    } // namespace ghex

} // namespace gridtools

#endif // INCLUDED_GHEX_COMMON_AWAIT_FUTURES_HPP

