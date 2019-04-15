#ifndef INCLUDED_REGULAR_DOMAIN_HPP
#define INCLUDED_REGULAR_DOMAIN_HPP


#include "domain_id_traits.hpp"

#include <utility>
#include <tuple>
#include <array>
#include <vector>
#include <map>
#include <algorithm>

//#include <iostream>
//#include <mpi.h>

namespace ghex {

namespace detail {

constexpr int int_pow(int base, int exp)
{
    return (exp==1 ? base : base*int_pow(base,exp-1));
}

template<typename F, typename Array, std::size_t... Is>
auto call_enumerated_impl(F&& f, const Array& a, std::index_sequence<Is...>)
{
    return f(a[Is]...);
}

template<int D, typename F, typename Array>
auto call_enumerated(F&& f, const Array& a)
{
    //return call_enumerated_impl(std::forward<F>(f), a, std::make_index_sequence<std::tuple_size<Array>::value>());
    return call_enumerated_impl(std::forward<F>(f), a, std::make_index_sequence<D>{});
}

template<typename I, int D>
struct coordinate
{
    using array_type     = std::array<I,D>;
    using iterator       = typename array_type::iterator;
    using const_iterator = typename array_type::const_iterator;
    using dimension      = std::integral_constant<int,D>;

    static constexpr int size() noexcept { return D; }

    array_type m_coord;

    coordinate() noexcept = default;
    coordinate(I scalar) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i]=scalar;
    }

    const I& operator[](int i) const noexcept { return m_coord[i]; }
    I& operator[](int i) noexcept { return m_coord[i]; }

    iterator begin() noexcept { return m_coord.begin(); }
    const_iterator begin() const noexcept { return m_coord.cbegin(); }
    const_iterator cbegin() const noexcept { return m_coord.cbegin(); }

    iterator end() noexcept { return m_coord.end(); }
    const_iterator end() const noexcept { return m_coord.cend(); }
    const_iterator cend() const noexcept { return m_coord.cend(); }

    coordinate& operator+=(const coordinate& c) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i] += c.m_coord[i];
        return *this;
    }
    coordinate& operator+=(I scalar) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i] += scalar;
        return *this;
    }
    coordinate& operator-=(const coordinate& c) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i] -= c.m_coord[i];
        return *this;
    }
    coordinate& operator-=(I scalar) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i] -= scalar;
        return *this;
    }
    /*coordinate operator%=(const coordinate& c) noexcept
        for (int i=0; i<D; ++i) m_coord[i] %= c.m_coord[i];
        return *this;
    }
    coordinate& operator%=(I scalar) noexcept
    {
        for (int i=0; i<D; ++i) m_coord[i] %= scalar;
        return *this;
    }*/
};

template<typename I, int D>
coordinate<I,D> operator+(coordinate<I,D> l, const coordinate<I,D>& r) noexcept
{
    return std::move(l+=r);
}
template<typename I, int D>
coordinate<I,D> operator+(coordinate<I,D> l, I scalar) noexcept
{
    return std::move(l+=scalar);
}
template<typename I, int D>
coordinate<I,D> operator-(coordinate<I,D> l, const coordinate<I,D>& r) noexcept
{
    return std::move(l-=r);
}
template<typename I, int D>
coordinate<I,D> operator-(coordinate<I,D> l, I scalar) noexcept
{
    return std::move(l-=scalar);
}
template<typename I, int D>
coordinate<I,D> operator%(coordinate<I,D> l, const coordinate<I,D>& r) noexcept
{
    for (int i=0; i<D; ++i) l[i] = l[i]%r[i];
    return std::move(l);
}
template<typename I, int D>
coordinate<I,D> operator%(coordinate<I,D> l, I scalar) noexcept
{
    for (int i=0; i<D; ++i) l[i] = l[i]%scalar;
    return std::move(l);
    //return std::move(l%=scalar);
}

} // namespace detail

template<
    int D,
    typename LocalIndex, 
    typename GlobalCellID, 
    typename DomainID,
    typename Layout = void> 
    //typename Partition>
class regular_domain
{
public: // public member types

    using local_index_type    = LocalIndex;
    using local_coord_type    = detail::coordinate<local_index_type,D>;//std::array<local_index_type,D>
    using dimension           = std::integral_constant<int,D>;
    using global_cell_id_type = GlobalCellID;
    using domain_id_type      = DomainID;
    //using partition           = Partition;
    
//private: // private member types

    struct local_box_type
    {
        local_coord_type m_first;
        local_coord_type m_last;

        local_coord_type& first() noexcept { return m_first; }
        const local_coord_type& first() const noexcept { return m_first; }
        local_coord_type& last() noexcept { return m_last; }
        const local_coord_type& last() const noexcept { return m_last; }
    };
    
    struct global_box_type
    {
        local_box_type m_local_box;
        global_cell_id_type m_first;
        global_cell_id_type m_last;

        template<typename Map>
        global_box_type(const local_box_type& local_box, Map&& m)
        :   m_local_box{local_box},
            m_first(detail::call_enumerated<D>(m, local_box.first())),
            m_last(detail::call_enumerated<D>(m, local_box.last()))
        {}

        /*template<typename Map>
        global_box_type(const local_coord_type& first, const local_coord_type& last, Map&& m)
        :   m_local_box{first,last},
            m_first(m(first)), detail::
            m_last(m(last))
        {}*/

        global_box_type(const global_box_type&) noexcept = default;
        global_box_type(global_box_type&&) noexcept = default;
        global_box_type& operator=(const global_box_type&) noexcept = default;
        global_box_type& operator=(global_box_type&&) noexcept = default;

        local_coord_type& first() noexcept { return m_local_box.first(); }
        const local_coord_type& first() const noexcept { return m_local_box.first(); }
        local_coord_type& last() noexcept { return m_local_box.last(); }
        const local_coord_type& last() const noexcept { return m_local_box.last(); }

        std::size_t size() const 
        {
            const auto diff = m_local_box.last()-m_local_box.first()+1;
            std::size_t res = diff[0];
            for (int i=1; i<D; ++i) res*=diff[i];
            return res;
        }

        bool operator<(const global_box_type& other) const noexcept
        {
            if (m_first < other.m_first) 
                return true;
            else if (m_first > other.m_first)
                return false;
            else if (m_last < other.m_last)
                return true;
            else 
                return false;
        }
    };

    using map_type = std::map<domain_id_type, std::pair<std::size_t,std::vector<global_box_type>>>;
    //using iterator = typename map_type::iterator;
    //using const_iterator = typename map_type::const_iterator;

private: // members

    local_coord_type m_origin;
    local_coord_type m_extent;
    local_coord_type m_halo_left;
    local_coord_type m_halo_right;

    map_type m_inner_map;
    map_type m_outer_map;

public: // ctors

    template<typename IndexRange, typename HaloRange, typename GIDMap, typename DIDMap>
    regular_domain(IndexRange&& origin, IndexRange&& extent, HaloRange&& halo_left, HaloRange&& halo_right, GIDMap&& gmap, DIDMap&& dmap)
    {
        std::copy(std::begin(origin), std::end(origin), std::begin(m_origin));
        std::copy(std::begin(extent), std::end(extent), std::begin(m_extent));
        std::copy(std::begin(halo_left), std::end(halo_left), std::begin(m_halo_left));
        std::copy(std::begin(halo_right), std::end(halo_right), std::begin(m_halo_right));

        // generate all possible iteration spaces / boxes
        // and sort them

        // 0,           halo_left,          halo_left+extent
        // halo_left-1, halo_left+extent-1, halo_left+extent+halo_right-1
        std::array<local_box_type, 3> outer_spaces{
            local_box_type{ m_origin-m_halo_left,   m_origin-1 },
            local_box_type{ m_origin,               m_origin+m_extent-1 },
            local_box_type{ m_origin+m_extent,      m_origin+m_extent+m_halo_right-1 }
        };

        // generate outer boxes
        auto local_outer_spaces = compute_spaces<local_box_type>(outer_spaces);

        // modulo operation to generate inner spaces
        std::remove_reference_t<decltype(local_outer_spaces)> local_inner_spaces;
        local_inner_spaces.reserve(local_outer_spaces.size());
        for (const auto& s : local_outer_spaces)
            local_inner_spaces.push_back(
                local_box_type
                {
                    (s.first()-m_origin+m_extent)%m_extent+m_origin,
                    (s.last() -m_origin+m_extent)%m_extent+m_origin
                }
            );

        // fill map and order
        for (int j=0; j<static_cast<int>(local_outer_spaces.size()); ++j)
        {
            if (j==(detail::int_pow(3,D)/2)) continue;
            const auto domain_id_outer = dmap( detail::call_enumerated<D>(gmap, local_outer_spaces[j].first()) );
            bool empty = false;
            for (int i=0; i<D; ++i)
            {
                if (local_outer_spaces[j].last()[i] < local_outer_spaces[j].first()[i])
                {
                    empty = true;
                    break;
                }
            }
            if (empty) continue;
            auto diff = local_outer_spaces[j].first() - local_inner_spaces[j].first();
            for (int i=0; i<D; ++i)
            {
                const auto dp = local_inner_spaces[j].last()[i]-local_inner_spaces[j].first()[i]+1;
                const auto dm = local_inner_spaces[j].first()[i]-local_inner_spaces[j].last()[i]-1;
                diff[i] = diff[i] != static_cast<local_index_type>(0) ? 
                    (diff[i] > static_cast<local_index_type>(0) ? dm : dp) :
                    static_cast<local_index_type>(0);
            }
            const auto neighbor_look_up = local_inner_spaces[j].first() + diff;
            const auto domain_id_inner = dmap( detail::call_enumerated<D>(gmap, neighbor_look_up) );
            if (domain_id_inner != domain_id_traits<domain_id_type>::invalid)
            {
                m_inner_map[domain_id_inner].second.push_back( global_box_type(local_inner_spaces[j], gmap) );
            }
            if (domain_id_outer != domain_id_traits<domain_id_type>::invalid)
            {
                m_outer_map[domain_id_outer].second.push_back( global_box_type(local_outer_spaces[j], gmap) );
            }
        }

        for (auto& p : m_inner_map)
        {
            std::sort(p.second.second.begin(), p.second.second.end());
            // TODO fuse adjacent boxes!
            std::size_t s=0;
            for (auto& x : p.second.second)
            {
                s += x.size();
            }
            p.second.first = s;
        }
        for (auto& p : m_outer_map)
        {
            std::sort(p.second.second.begin(), p.second.second.end());
            // TODO fuse adjacent boxes!
            std::size_t s=0;
            for (auto& x : p.second.second)
            {
                s += x.size();
            }
            p.second.first = s;
        }
    }

    const map_type& inner() const { return m_inner_map; }
    const map_type& outer() const { return m_outer_map; }

private:


    template<typename Box, typename Spaces>
    std::vector<Box> compute_spaces(const Spaces& spaces)
    {
        std::vector<Box> x;
        x.reserve(detail::int_pow(3,D));
        Box b;
        compute_spaces<Box>(0, spaces, b, x);
        return std::move(x);
    }

    template<typename Box, typename Spaces, typename Container>
    void compute_spaces(int d, const Spaces& spaces, Box& current_box, Container& c)//, GIDMap&& gmap, DIDMap&& dmap)
    {
        if (d==D)
        {
             c.push_back(current_box);
        }
        else
        {
            current_box.first()[d] = spaces[0].first()[d];
            current_box.last()[d]  = spaces[0].last()[d];
            compute_spaces(d+1, spaces, current_box, c);//, std::forward<GIDMap>(gmap), std::forward<DIDMap>(dmap));

            current_box.first()[d] = spaces[1].first()[d];
            current_box.last()[d]  = spaces[1].last()[d];
            compute_spaces(d+1, spaces, current_box, c);//, std::forward<GIDMap>(gmap), std::forward<DIDMap>(dmap));
            
            current_box.first()[d] = spaces[2].first()[d];
            current_box.last()[d]  = spaces[2].last()[d];
            compute_spaces(d+1, spaces, current_box, c);//, std::forward<GIDMap>(gmap), std::forward<DIDMap>(dmap));
        }
    }
    /*()
    {
        
    }*/

};


} // namespace ghex


#endif /* INCLUDED_REGULAR_DOMAIN_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

