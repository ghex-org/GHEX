#pragma once

namespace gridtools {
    class halo_range {
        int m_begin, m_end;

    public:
        halo_range() : m_begin{-1}, m_end{-2} {}

        constexpr halo_range(int begin, int end)
            : m_begin(begin)
            , m_end(end)
        {}

        constexpr int begin() const { return m_begin; }
        constexpr int end() const { return m_end; }
    };

    constexpr int begin(halo_range const hr) { return hr.begin(); }
    constexpr int end(halo_range const hr) { return hr.end(); }
}
