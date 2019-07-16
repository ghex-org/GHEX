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
#ifndef INCLUDED_ACCUMULATOR_HPP
#define INCLUDED_ACCUMULATOR_HPP

namespace gridtools {
    
    namespace detail {
        struct accumulator_impl;
    }

    class accumulator
    {
    private: // members
        detail::accumulator_impl* m_impl;
        bool m_moved = false;

    public: // ctors
        accumulator();
        accumulator(const accumulator&);
        accumulator(accumulator&&);
        ~accumulator();

    public: // member functions
        void operator()(double);
        double mean();
        double stdev();
        double min();
        double max();
    };
}

#endif /* INCLUDED_ACCUMULATOR_HPP */

