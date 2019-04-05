#ifndef INCLUDED_COMMUNICATION_OBJECT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_HPP

#include "util.hpp"
#include "regular_domain.hpp"
#include <functional>
#include <mpi.h>


namespace ghex {

enum class protocol : int
{
    mpi_async
};


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

    using domain_type = regular_domain<D,LocalIndex,GlobalCellID,DomainID>;
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
        
    }

    template<typename BufferMap, typename Field>
    void unpack(const BufferMap& buffer_map, Field& field) const
    {
        
    }
};


template<typename Allocator = std::allocator<char>, typename... CommunicationObjects>
class exchange
{ };

template<typename Allocator, typename... Domain, typename... T, typename... Backend>
class exchange<Allocator, communication_object<Domain,T,protocol::mpi_async,Backend>...>
{ 
private:

    using allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<char>;

    using communication_objects_tuple = std::tuple<communication_object<Domain,T,protocol::mpi_async,Backend>&...>;

    using size = std::integral_constant<std::size_t, sizeof...(T)>;

    using value_types = std::tuple<T...>;

    using max_alignment_t = typename detail::ct_reduce<
        detail::ct_max,
        std::integral_constant<std::size_t, 0>,
        std::integral_constant<std::size_t,alignof(T)>... 
    >::type;

    struct ledger
    {
        std::size_t total_size = 0;
        std::array<std::size_t, size::value> offset;
    };

    using ledger_map_type = std::map<int, ledger>;    
    using pack_type = std::array< std::map<int, void*>, size::value>; 

    communication_objects_tuple m_cos;
    allocator_t                 m_alloc;
    ledger_map_type             m_inner_ledger_map;
    pack_type                   m_inner_pack;
    std::vector<std::pair<char*,std::size_t>> m_inner_memory;
    std::vector<std::pair<char*,std::size_t>> m_outer_memory;

public:

    template<typename... CommunicationObjects>
    exchange(MPI_Comm comm, const Allocator& alloc, CommunicationObjects&... cos)
    :   m_cos{cos...},
        m_alloc{ alloc }
    {
        int i = 0;
        detail::for_each(
            m_cos, [&i, this](const auto& co) 
            {
                using co_t    = std::remove_reference_t<decltype(co)>;
                using value_t = typename co_t::value_type; 
                std::cout << co.inner().size() << std::endl; 
                for (const auto& p : co.inner())
                {
                    auto& l = this->m_inner_ledger_map[p.first];
                    l.offset[i]   = l.total_size;
                    l.total_size += p.second.first*sizeof(value_t) + alignof(value_t)-1;
                }

                ++i;
            });
    }

    ~exchange()
    {
        for (auto p : m_inner_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, p.first, p.second);
        for (auto p : m_outer_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, p.first, p.second);
    }

    template<typename... Fields>
    void start(Fields&... fields)
    {
    }

    void wait()
    {
    }

private:


/*void* align(void* ptr, std::size_t alignment) noexcept
{
    std::size_t space = sizeof(T) + alignof(T);
    void* buffer = ptr;
    const auto alignment = alignof(T);
    const auto size = sizeof(T);
    return std::align(alignment, size, buffer, space);
}*/

};

} // namespace ghex


#endif /* INCLUDED_COMMUNICATION_OBJECT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

