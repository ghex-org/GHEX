#pragma once

namespace gridtools {
    class halo_range {
        int m_begin, m_end;

    public:
        halo_range(int begin, int end)
            : m_begin(begin)
            , m_end(end)
        {}

        int begin() const { return m_begin; }
        int end() const { return m_end; }
    };

    int begin(halo_range const hr) { return hr.begin(); }
    int end(halo_range const hr) { return hr.end(); }
}
