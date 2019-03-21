#ifndef INCLUDED_UNSTRUCTURED_GRID_HPP
#define INCLUDED_UNSTRUCTURED_GRID_HPP

#include <array>
#include <vector>
#include <set>
#include <algorithm>
#include <iosfwd>

namespace gridtools {

// begin(), end() -> id?
//
// principal_neighbor(id, fun)  -> fun(id_e, PE)
//
// neighbor<I>(id, fun) -> fun(id_e, PE)
// I is level of neighborhood, i.e. 0 <=> principal neighbor
//
// operator()(id, k) -> T&
//
// 


static std::array<std::pair<int,int>,64> coordinate_lu{
    std::pair<int,int>{0,0}, // 0
    {1,0}, // 1
    {1,1}, // 2
    {0,1}, // 3
    {0,2}, // 4
    {0,3}, // 5
    {1,3}, // 6
    {1,2}, // 7
    {2,2}, // 8
    {2,3}, // 9
    {3,3}, //10
    {3,2}, //11
    {3,1}, //12
    {2,1}, //13
    {2,0}, //14
    {3,0}, //15
    {4,0}, //16
    {4,1}, //17
    {5,1}, //18
    {5,0}, //19
    {6,0}, //20
    {7,0}, //21
    {7,1}, //22
    {6,1}, //23
    {6,2}, //24
    {7,2}, //25
    {7,3}, //26
    {6,3}, //27
    {5,3}, //28
    {5,2}, //29
    {4,2}, //30
    {4,3}, //31
    {4,4}, //32
    {4,5}, //33
    {5,5}, //34
    {5,4}, //35
    {6,4}, //36
    {7,4}, //37
    {7,5}, //38
    {6,5}, //39
    {6,6}, //40
    {7,6}, //41
    {7,7}, //42
    {6,7}, //43
    {5,7}, //44
    {5,6}, //45
    {4,6}, //46
    {4,7}, //47
    {3,7}, //48
    {2,7}, //49
    {2,6}, //50
    {3,6}, //51
    {3,5}, //52
    {3,4}, //53
    {2,4}, //54
    {2,5}, //55
    {1,5}, //56
    {1,4}, //57
    {0,4}, //58
    {0,5}, //59
    {0,6}, //60
    {1,6}, //61
    {1,7}, //62
    {0,7}  //63
};

static std::array<int,64> index_lu{
     0, // 0,0
     1, // 1,0
    14, // 2,0
    15, // 3,0
    16, // 4,0
    19, // 5,0
    20, // 6,0
    21, // 7,0
     3, // 0,1
     2, // 1,1
    13, // 2,1
    12, // 3,1
    17, // 4,1
    18, // 5,1
    23, // 6,1
    22, // 7,1
     4, // 0,2
     7, // 1,2
     8, // 2,2
    11, // 3,2
    30, // 4,2
    29, // 5,2
    24, // 6,2
    25, // 7,2
     5, // 0,3
     6, // 1,3
     9, // 2,3
    10, // 3,3
    31, // 4,3
    28, // 5,3
    27, // 6,3
    26, // 7,3
    58, // 0,4
    57, // 1,4
    54, // 2,4
    53, // 3,4
    32, // 4,4
    35, // 5,4
    36, // 6,4
    37, // 7,4
    59, // 0,5
    56, // 1,5
    55, // 2,5
    52, // 3,5
    33, // 4,5
    34, // 5,5
    39, // 6,5
    38, // 7,5
    60, // 0,6
    61, // 1,6
    50, // 2,6
    51, // 3,6
    46, // 4,6
    45, // 5,6
    40, // 6,6
    41, // 7,6
    63, // 0,7
    62, // 1,7
    49, // 2,7
    48, // 3,7
    47, // 4,7
    44, // 5,7
    43, // 6,7
    42  // 7,7
};


//static std::array<int,6> decomposition{0,11,19,34,52,64};
static std::array<int,7> decomposition{0,11,19,30,42,52,64};

static int get_domain_id(int index)
{
    return (std::upper_bound(decomposition.begin(), decomposition.end(), index)-decomposition.begin())-1;
}


template<typename T>
struct local_unstructured_grid_data
{
    using id_type = int;
    using cell_id_type = int;

    int m_k;
    id_type m_id;
    cell_id_type m_begin;
    cell_id_type m_end;
    std::vector<T> m_data;
    std::vector<T> m_recv_data;
    std::vector<int> m_recv_index;
    std::vector<int> m_send_index;

    local_unstructured_grid_data(int k, int id) :
        m_k(k),
        m_id(id),
        m_begin(decomposition[id]),
        m_end(decomposition[id+1]),
        m_data(m_k*(m_end-m_begin))
    {
        std::set<int> neighbors;
        for (int i=m_begin; i<m_end; ++i)
        {
            const auto coord = coordinate_lu[i];
            const auto pn0 = ((coord.second+0+8)%8)*8 + ((coord.first-1+8)%8);
            const auto pn1 = ((coord.second-1+8)%8)*8 + ((coord.first+0+8)%8);
            const auto pn2 = ((coord.second+0+8)%8)*8 + ((coord.first+1+8)%8);
            const auto pn3 = ((coord.second+1+8)%8)*8 + ((coord.first+0+8)%8);

            if (index_lu[pn0] < m_begin || index_lu[pn0] >= m_end) neighbors.insert(index_lu[pn0]);
            if (index_lu[pn1] < m_begin || index_lu[pn1] >= m_end) neighbors.insert(index_lu[pn1]);
            if (index_lu[pn2] < m_begin || index_lu[pn2] >= m_end) neighbors.insert(index_lu[pn2]);
            if (index_lu[pn3] < m_begin || index_lu[pn3] >= m_end) neighbors.insert(index_lu[pn3]);
        }
        m_recv_data.resize(neighbors.size()*m_k);
        m_recv_index.insert(m_recv_index.begin(), neighbors.begin(), neighbors.end());

        neighbors.clear();
        for (auto r : m_recv_index)
        {
            const auto coord = coordinate_lu[r];
            const auto pn0 = ((coord.second+0+8)%8)*8 + ((coord.first-1+8)%8);
            const auto pn1 = ((coord.second-1+8)%8)*8 + ((coord.first+0+8)%8);
            const auto pn2 = ((coord.second+0+8)%8)*8 + ((coord.first+1+8)%8);
            const auto pn3 = ((coord.second+1+8)%8)*8 + ((coord.first+0+8)%8);

            if (index_lu[pn0] >= m_begin && index_lu[pn0] < m_end) neighbors.insert(index_lu[pn0]);
            if (index_lu[pn1] >= m_begin && index_lu[pn1] < m_end) neighbors.insert(index_lu[pn1]);
            if (index_lu[pn2] >= m_begin && index_lu[pn2] < m_end) neighbors.insert(index_lu[pn2]);
            if (index_lu[pn3] >= m_begin && index_lu[pn3] < m_end) neighbors.insert(index_lu[pn3]);
        }
        m_send_index.insert(m_send_index.begin(),neighbors.begin(), neighbors.end());

        // fill in some data
        for (int i=m_begin; i<m_end; ++i)
            for (int z=0; z<m_k; ++z)
            {
                const auto coord = coordinate_lu[i];
                this->operator()(i,z) = 1000*(m_id+1) + 100*coord.first + 10*coord.second + z;
            }
        for (auto r : m_recv_index)
            for (int z=0; z<m_k; ++z)
            {
                const auto coord = coordinate_lu[r];
                this->operator()(r,z) = 1000*(m_id+1) + 100*coord.first + 10*coord.second + z;
            }
    }

    bool is_inner(int index) const 
    {
        bool inner; 
        const auto idx = data_index(index,0,inner);
        return inner && idx >= 0;
    }

    bool is_outer(int index) const 
    {
        bool inner; 
        const auto idx = data_index(index,0,inner);
        return !inner && idx >= 0;
    }

    T& operator()(int index, int k)
    {
        bool inner;
        const auto idx = data_index(index,k,inner);
        if (idx < 0) throw std::runtime_error("out of bounds!");
        return inner ? m_data[idx] : m_recv_data[idx];
    }

    const T& operator()(int index, int k) const
    {
        bool inner;
        const auto idx = data_index(index,k,inner);
        if (idx < 0) throw std::runtime_error("out of bounds!");
        return inner ? m_data[idx] : m_recv_data[idx];
    }

    auto begin() { return m_data.begin(); }
    auto begin() const { return m_data.cbegin(); }
    auto cbegin() const { return m_data.cbegin(); }
    auto end() { return m_data.end(); }
    auto end() const { return m_data.cend(); }
    auto cend() const { return m_data.cend(); }

    template<int L, typename Fun>
    void visit_neighbors(int index, Fun&& f) 
    {
        visit_neighbors(index,std::forward<Fun>(f),std::integral_constant<int,L>());
    }

    template<typename Fun>
    void visit_neighbors(int index, Fun&& fun, std::integral_constant<int,0>)
    {
        const auto coord = coordinate_lu[index];
        {
            bool inner;
            const auto i0 = index_lu[((coord.second+0+8)%8)*8 + ((coord.first-1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second-1+8)%8)*8 + ((coord.first+0+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second+0+8)%8)*8 + ((coord.first+1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second+1+8)%8)*8 + ((coord.first+0+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
    }

    template<typename Fun>
    void visit_neighbors(int index, Fun&& fun, std::integral_constant<int,1>)
    {
        const auto coord = coordinate_lu[index];
        {
            bool inner;
            const auto i0 = index_lu[((coord.second-1+8)%8)*8 + ((coord.first-1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second-1+8)%8)*8 + ((coord.first+1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second+1+8)%8)*8 + ((coord.first+1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
        {
            bool inner;
            const auto i0 = index_lu[((coord.second+1+8)%8)*8 + ((coord.first-1+8)%8)];
            const auto idx = data_index(i0, 0, inner);
            if (idx >= 0) 
            {
                const auto r = get_domain_id(i0);
                if (inner)
                    for (int k=0; k<m_k; ++k)
                        fun(m_data[idx + k], r);
                else
                    for (int k=0; k<m_k; ++k)
                        fun(m_recv_data[idx + k], r);
            }
        }
    }


public: // print

    template< class CharT, class Traits = std::char_traits<CharT>> 
    friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const local_unstructured_grid_data& g)
    {
        os << "\n";
        for (int x=0; x<8; ++x)
        {
            for (int y=0; y<8; ++y)
            {
                const auto index = index_lu[y*8 + x];
                const auto r = get_domain_id(index);
                if (g.is_inner(index)) 
                    //os << "\033[1;3"<< r+1 <<"m" 
                    os << "\033[1m\033[7;3"<< r+1 <<"m" 
                       << g(index, 0)
                       << "\033[0m"
                       << " ";
                else if (g.is_outer(index))
                    //os << "\033[1m\033[7;3"<< r+1 <<"m" 
                    os << "\033[1;3"<< r+1 <<"m" 
                       << g(index,0)  
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

    int data_index(int index, int k, bool& inner) const
    {
        if (index < m_begin || index >= m_end)
        {
            inner = false;
            auto it = std::lower_bound(m_recv_index.begin(), m_recv_index.end(), index);
            if (it != m_recv_index.end() && *it == index)
            {
                return (it-m_recv_index.begin())*m_k+k;
            }
            else return -1;
        }
        else
        {
            inner = true;
            return (index-m_begin)*m_k+k;
        }
    }

};




template<typename T>
struct local_structured_grid_data
{


};



}




#endif /* INCLUDED_UNSTRUCTURED_GRID_HPP */
