#ifndef INCLUDED_STRUCTURED_GRID_HPP
#define INCLUDED_STRUCTURED_GRID_HPP

#include <iostream>
#include <array>
#include <vector>

namespace gridtools {

static std::array< std::array<int, 4>, 6> structured_decomposition{ 
    std::array<int,4>{0,0,4,3}, 
    {4,0,4,3}, 
    {0,3,4,3}, 
    {4,3,4,3}, 
    {0,6,4,2}, 
    {4,6,4,2}};

static int get_structured_domain_id(int index)
{
    const auto y = index/8;
    const auto x = index - (y*8);
    // crude linear searc
    int q=0;
    for (const auto d : structured_decomposition)
    {
        if (x >= d[0] && x<d[0]+d[2] &&
            y >= d[1] && y<d[1]+d[3])
        {
            return q;
        }
        ++q;
    }
    return q;
}

template<typename T>
struct local_structured_grid_data
{
    struct _local_cell_id_t : std::array<int,2>
    { 
        _local_cell_id_t() noexcept = default;
        _local_cell_id_t(int x, int y) noexcept 
        {
            this->operator[](0) = x;
            this->operator[](1) = y;
        }
        bool operator<(const _local_cell_id_t& other) const
        {
            return 
                ((this->operator[](0) < other[0]) ? 
                    true :
                    ((this->operator[](0) > other[0]) ?
                        false :
                        ((this->operator[](1) < other[1]) ?
                            true :
                            false
                        )
                    ) 
                );
        }
    };

    using local_cell_id_t  = _local_cell_id_t;
    using global_cell_id_t = int;
    using extent_t         = int;
    using domain_id_t      = int;

    extent_t m_k;
    domain_id_t m_id;
    local_cell_id_t m_begin;
    local_cell_id_t m_end;
    std::vector<T> m_data;

    local_structured_grid_data(extent_t k, domain_id_t id) :
       m_k(k),
       m_id(id),
       m_begin(0,0),
       m_end(structured_decomposition[m_id][2]+2, structured_decomposition[m_id][3]+2),
       m_data(m_end[0]*m_end[1]*m_k)
    {
        // fill in some data
        for (auto y=m_begin[1]; y<m_end[1]; ++y)
        for (auto x=m_begin[0]; x<m_end[0]; ++x)
        for (extent_t z=0; z<m_k; ++z)
        {
            const auto gx = ((structured_decomposition[m_id][0]-1+x)+8)%8;
            const auto gy = ((structured_decomposition[m_id][1]-1+y)+8)%8;
            this->operator()(local_cell_id_t(x,y), z) = 1000*(m_id+1) + 100*(gx) + 10*(gy) + z;
        }
    }

    global_cell_id_t global_cell_id(local_cell_id_t id) const
    {
        const auto x = ((structured_decomposition[m_id][0]-1+id[0])+8)%8;
        const auto y = ((structured_decomposition[m_id][1]-1+id[1])+8)%8;
        return y*8+x;
    }

    T& operator()(local_cell_id_t id, extent_t k)
    {
        return m_data[data_index(id,k)];
    }

    const T& operator()(local_cell_id_t id, extent_t k) const
    {
        return m_data[data_index(id,k)];
    }

public: // print

    template< class CharT, class Traits = std::char_traits<CharT>> 
    friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const local_structured_grid_data& g)
    {
        os << "\n";
        for (int x=0; x<8; ++x)
        {
            for (int y=0; y<8; ++y)
            {
                const global_cell_id_t index = y*8 + x;
                const auto r = get_structured_domain_id(index);
                local_cell_id_t loc_id; 
                if (g.is_inner(index)) 
                {
                    const auto xl = x-structured_decomposition[g.m_id][0]+1;
                    const auto yl = y-structured_decomposition[g.m_id][1]+1;
                    //os << "\033[1;3"<< r+1 <<"m" 
                    os << "\033[1m\033[7;3"<< r+1 <<"m" 
                       << g({xl,yl}, 0)
                       << "\033[0m"
                       << " ";
                }
                else if (g.is_outer(index, loc_id))
                    //os << "\033[1m\033[7;3"<< r+1 <<"m" 
                    os << "\033[1;3"<< r+1 <<"m" 
                       //<< g(index,0)
                       << g(loc_id,0)
                       //<< "xxxx"  
                       << "\033[0m" 
                       << " ";
                else 
                    //os << "  x  ";
                    os << "  ·  ";
            }
            os << "\n";
        }
        os << "\n";
        return os;
    }

private:

    int data_index(local_cell_id_t id, extent_t k) const
    {
        return id[1]*(m_end[0]*m_k) + id[0]*(m_k) + k;
    }

    bool is_inner(global_cell_id_t id) const
    {
        const auto y = id/8;
        const auto x = id-y*8;
        if (x>=structured_decomposition[m_id][0] &&
            x <structured_decomposition[m_id][0]+structured_decomposition[m_id][2] &&
            y>=structured_decomposition[m_id][1] &&
            y <structured_decomposition[m_id][1]+structured_decomposition[m_id][3])
        return true;
        else return false;
    }

    bool is_outer(global_cell_id_t id, local_cell_id_t& loc_id) const
    {
        const auto sw = m_begin;
        const auto ne = local_cell_id_t{m_end[0]-1,m_end[1]-1};
        const auto se = local_cell_id_t{ne[0],sw[1]};
        const auto nw = local_cell_id_t{sw[0],ne[1]};

        if (id == global_cell_id(sw))
        {
            loc_id = sw;
            return true;
        }
        if (id == global_cell_id(se))
        {
            loc_id = se;
            return true;
        }
        if (id == global_cell_id(ne))
        {
            loc_id = ne;
            return true;
        }
        if (id == global_cell_id(nw))
        {
            loc_id = nw;
            return true;
        }

        const auto y_s = global_cell_id(sw)/8;
        const auto y_n = global_cell_id(ne)/8;
        const auto x_w = global_cell_id(sw) - y_s*8;
        const auto x_e = global_cell_id(ne) - y_n*8;

        const auto y = id/8;
        const auto x = id-y*8;

        const bool x_inner = x>=structured_decomposition[m_id][0] && x <structured_decomposition[m_id][0]+structured_decomposition[m_id][2];
        const bool y_inner = y>=structured_decomposition[m_id][1] && y <structured_decomposition[m_id][1]+structured_decomposition[m_id][3];

        if (y==y_s && x_inner) 
        {
            loc_id = local_cell_id_t{x-structured_decomposition[m_id][0]+1, sw[2]};
            return true;
        }
        if (y==y_n && x_inner)
        {
            loc_id = local_cell_id_t{x-structured_decomposition[m_id][0]+1, nw[2]};
            return true;
        }
        if (x==x_e && y_inner)
        {
            loc_id = local_cell_id_t{se[0], y-structured_decomposition[m_id][1]+1};
            return true;
        }
        if (x==x_w && y_inner)
        {
            loc_id = local_cell_id_t{sw[0], y-structured_decomposition[m_id][1]+1};
            return true;
        }
        return false;
    }
       
};



}

#endif /* INCLUDED_STRUCTURED_GRID_HPP */

