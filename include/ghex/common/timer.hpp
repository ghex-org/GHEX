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
#ifndef INCLUDED_GHEX_COMMON_TIMER_HPP
#define INCLUDED_GHEX_COMMON_TIMER_HPP

#include "./accumulator.hpp"
#include <chrono>

namespace gridtools {

    namespace ghex {

        /** @brief timer with built-in statistics */
        class timer : public accumulator
        {
        private: // member types
            using base = accumulator;
            using clock_type = std::chrono::high_resolution_clock;
            using time_point = typename clock_type::time_point;

        private: // members
            time_point m_time_point = clock_type::now();

        public: // ctors
            timer() = default;
            timer(const base& b) : base(b) {}
            timer(base&& b) : base(std::move(b)) {}
            timer(const timer&) noexcept = default;
            timer(timer&&) noexcept = default;
            timer& operator=(const timer&) noexcept = default;
            timer& operator=(timer&&) noexcept= default;

        public: // time functions

            /** @brief start timings */
            inline void tic() noexcept
            {
                m_time_point = clock_type::now();
            }
            /** @brief stop timings */
            inline double stoc() noexcept
            {
                return std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - m_time_point).count();
            }

            /** @brief stop timings */
            inline double toc() noexcept
            {
                const auto t = std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - m_time_point).count();
                this->operator()( t );
                return t;
            }

            /** @brief stop timings, verbose: print measured time */
            inline void vtoc() noexcept
            {
		double t = std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - m_time_point).count();
		std::cout << "time:      " << t/1000000 << "s\n";
            }

            /** @brief stop timings, verbose: print measured time and bandwidth */
            inline void vtoc(const char* header, long bytes) noexcept
            {
		double t = std::chrono::duration_cast<std::chrono::microseconds>(clock_type::now() - m_time_point).count();
		std::cout << header << " MB/s:      " << bytes/t << "\n";
            }

            /** @brief stop and start another timing period */
            inline void toc_tic() noexcept
            {
                auto t2 = clock_type::now();
                this->operator()(
                    std::chrono::duration_cast<std::chrono::microseconds>(t2 - m_time_point).count());
                m_time_point = t2;
            }
        };

        /** @brief all-reduce timers over the MPI group defined by the communicator
          * @param t timer local to each rank
          * @param comm MPI communicator
          * @return combined timer incorporating statistics over all timings */
        timer reduce(const timer& t, MPI_Comm comm)
        {
            return reduce(static_cast<accumulator>(t), comm);
        }

    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_COMMON_TIMER_HPP */

