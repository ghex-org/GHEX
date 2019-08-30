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
#ifndef INCLUDED_GHEX_TIMER_HPP
#define INCLUDED_GHEX_TIMER_HPP

#include "./accumulator.hpp"
#include <chrono>

namespace gridtools {
namespace ghex {

class timer : public accumulator
{
    using base = accumulator;
    using clock_type = std::chrono::high_resolution_clock;
    //using rep = typename clock_type::rep;
    using time_point = typename clock_type::time_point;

    time_point m_time_point = clock_type::now();
public:
    timer() noexcept = default;
    timer(const base& b) : base(b) {}
    timer(base&& b) : base(std::move(b)) {}
    timer(const timer&) noexcept = default;
    timer(timer&&) noexcept = default;
	timer& operator=(const timer&) noexcept = default;
	timer& operator=(timer&&) noexcept= default;

    inline void tic() noexcept
    {
        m_time_point = clock_type::now();
    }
    inline void toc() noexcept
    {
        this->operator()( std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - m_time_point).count() );
    }
    inline void toc_tic() noexcept
    {
        toc();
        tic();
    }
};


timer reduce(const timer& t, MPI_Comm comm)
{
    return reduce(static_cast<accumulator>(t), comm);
}

} // namespace ghex
} // namespace gridtools


#endif /* INCLUDED_GHEX_TIMER_HPP */

