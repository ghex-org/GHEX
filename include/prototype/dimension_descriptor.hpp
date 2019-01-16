/** @file A new halo_decsriptor class that does not include the total
    lenght (i,e,. stride)
*/

#pragma once

#include <gridtools/common/array.hpp>
#include <prototype/halo_range.hpp>

namespace gridtools {
    class dimension_descriptor {
        int m_minus;
        int m_plus;
        int m_begin;
        int m_end; /* inclusive to not brake the old API */

        array<halo_range, 3> m_inner_ranges;
        array<halo_range, 3> m_outer_ranges;
    public:

        constexpr dimension_descriptor(int m, int p, int b, int e)
            : m_minus(m)
            , m_plus(p)
            , m_begin(b)
            , m_end(e)
            , m_outer_ranges{halo_range(m_begin - m_minus, m_begin), halo_range(m_begin, m_end + 1), halo_range(m_end + 1, m_end+m_plus + 1)}
            , m_inner_ranges{halo_range(m_begin, m_plus), halo_range(m_begin, m_plus), halo_range(m_end - m_minus + 1, m_end + 1)}
        {}

        constexpr halo_range inner_range(int i) const {
            return m_inner_ranges[i+1];
        }

        constexpr halo_range outer_range(int i) const {
            return m_outer_ranges[i+1];
        }

    };

}
