#ifndef INCLUDED_COMMUNICATION_OBJECT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_HPP

#include "util.hpp"
#include "regular_domain.hpp"
#include "protocol.hpp"
//#include <functional>
//#include <mpi.h>
//#include <memory>

#include <gridtools/common/layout_map.hpp>

namespace ghex {

namespace detail {

template<int D, int I, typename Layout=void>
struct for_loop
{
    using idx = std::integral_constant<int,D-I>;

    template<typename Func, typename Array>
    inline static void apply(Func&& f, Array&& first, Array&& last) noexcept
    {
        for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
        {
            std::remove_const_t<std::remove_reference_t<Array>> x{};
            x[idx::value] = i;
            for_loop<D,I-1,Layout>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
        }
    }

    template<typename Func, typename Array, typename Array2>
    inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& y) noexcept
    {
        for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
        {
            std::remove_const_t<std::remove_reference_t<Array2>> x{y};
            x[idx::value] = i;
            for_loop<D,I-1,Layout>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
        }
    }
};

template<int D,typename Layout>
struct for_loop<D,0,Layout>
{
    
    template<typename Func, typename Array, typename Array2>
    inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& x) noexcept
    {
        apply_impl(std::forward<Func>(f), std::forward<Array2>(x), std::make_index_sequence<D>{});
    }

    template<typename Func, typename Array, std::size_t... Is>
    inline static void apply_impl(Func&& f, Array&& x, std::index_sequence<Is...>)
    {
        f(x[Is]...);
    }
};


template<int D, int I, int... Args>
struct for_loop<D,I,gridtools::layout_map<Args...>>
{
    using layout_t = gridtools::layout_map<Args...>;
    using idx = std::integral_constant<int, layout_t::template find<D-I>()>;

    template<typename Func, typename Array>
    inline static void apply(Func&& f, Array&& first, Array&& last) noexcept
    {
        for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
        {
            std::remove_const_t<std::remove_reference_t<Array>> x{};
            x[idx::value] = i;
            for_loop<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
        }
    }

    template<typename Func, typename Array, typename Array2>
    inline static void apply(Func&& f, Array&& first, Array&& last, Array2&& y) noexcept
    {
        for(auto i=first[idx::value]; i<=last[idx::value]; ++i)
        {
            std::remove_const_t<std::remove_reference_t<Array2>> x{y};
            x[idx::value] = i;
            for_loop<D,I-1,layout_t>::apply(std::forward<Func>(f), std::forward<Array>(first), std::forward<Array>(last), x);
        }
    }

};

template<int D, int... Args>
struct for_loop<D,0,gridtools::layout_map<Args...>> : for_loop<D,0,void> {};

} // namespace detail


template<typename Domain, typename T, protocol P, typename Backend = void>
class communication_object
{};


template<
    int D,
    typename LocalIndex, 
    typename GlobalCellID, 
    typename DomainID,
    typename Layout,
    typename T,
    typename Backend
> 
class communication_object<regular_domain<D,LocalIndex,GlobalCellID,DomainID,Layout>, T, protocol::mpi_async, Backend>
{
public: // member types

    using domain_type = regular_domain<D,LocalIndex,GlobalCellID,DomainID,Layout>;
    using value_type  = T;
    using box_type    = typename domain_type::global_box_type;
    using layout_type = Layout;
    using map_type = std::map<int, std::pair<std::size_t, std::vector<box_type>>>;

private:

    map_type m_inner;
    map_type m_outer;

public:

    template<typename RMap>
    communication_object(const domain_type& domain, RMap&& m)
    {
        for (const auto& p : domain.inner())
        {
            const int rank = m(p.first);
            auto it = m_inner.find(rank);
            if (it == m_inner.end())
            {
                auto& _inner = m_inner[rank];
                _inner.second.insert(_inner.second.end(),p.second.second.begin(), p.second.second.end());
                _inner.first = p.second.first;
            }
            else
            {
                it->second.second.insert(it->second.second.end(),p.second.second.begin(), p.second.second.end());
                it->second.first += p.second.first;
            }
        }
        for (const auto& p : domain.outer())
        {
            const int rank = m(p.first);
            auto it = m_outer.find(rank);
            if (it == m_outer.end())
            {
                auto& _outer = m_outer[rank];
                _outer.second.insert(_outer.second.end(),p.second.second.begin(), p.second.second.end());
                _outer.first = p.second.first;
            }
            else
            {
                it->second.second.insert(it->second.second.end(),p.second.second.begin(), p.second.second.end());
                it->second.first += p.second.first;
            }
        }
    }

    const map_type& inner() const noexcept { return m_inner; }
    const map_type& outer() const noexcept { return m_outer; }

    template<typename BufferMap, typename Field>
    void pack(BufferMap& buffer_map, const Field& field) const
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            std::size_t i = 0;
            //for (const auto& box : m_inner[rank].second)
            for (const auto& box : m_inner.at(rank).second)
            {
                detail::for_loop<D,D,layout_type>::apply(
                    [buffer,&i,&field](auto... is)
                    {
                        buffer[i++] = field(is...); 
                    }, 
                    box.first(), 
                    box.last()
                );
                /*for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                    buffer[i++] = field(x,y,z);*/
            }
        }
    }

    template<typename BufferMap, typename Field>
    void unpack(const BufferMap& buffer_map, Field& field) const
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            std::size_t i = 0;
            //for (const auto& box : m_outer[rank].second)
            for (const auto& box : m_outer.at(rank).second)
            {
                detail::for_loop<D,D,layout_type>::apply(
                    [buffer,&i,&field](auto... is)
                    {
                        field(is...) = buffer[i++]; 
                    }, 
                    box.first(), 
                    box.last()
                );
                /*for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                    field(x,y,z) = buffer[i++];*/
            }
        }
    }
};



} // namespace ghex


#endif /* INCLUDED_COMMUNICATION_OBJECT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

