#ifndef INCLUDED_COMMUNICATION_OBJECT_RT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_RT_HPP

#include "communication_object.hpp"

namespace ghex {

template<typename Domain, protocol P, typename Backend = void>
class communication_object_rt
{};


template<
    int D,
    typename LocalIndex, 
    typename GlobalCellID, 
    typename DomainID,
    typename Layout,
    typename Backend
> 
class communication_object_rt<regular_domain<D,LocalIndex,GlobalCellID,DomainID,Layout>, protocol::mpi_async, Backend>
{
public: // member types

    using domain_type = regular_domain<D,LocalIndex,GlobalCellID,DomainID,Layout>;
    using box_type    = typename domain_type::global_box_type;
    using layout_type = Layout;
    using map_type = std::map<int, std::pair<std::size_t, std::vector<box_type>>>;

private:

    map_type m_inner;
    map_type m_outer;

public:

    template<typename RMap>
    communication_object_rt(const domain_type& domain, RMap&& m)
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

    template<typename U, typename BufferMap, typename Field>
    void pack(BufferMap& buffer_map, const Field& field) const
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<U*>(p.second);
            std::size_t j = 0;
            for (const auto& box : m_inner.at(rank).second)
            {
                // this is a bit cheating
                detail::for_loop_simple<D,D,layout_type>::apply(
                    [buffer,&j,&field](std::size_t offset, std::size_t j_offset)
                    {
                        buffer[j+j_offset] = field.ptr[offset]; 
                    }, 
                    box.first(), 
                    box.last(),
                    field.ext 
                );
                j+= box.size();
            }
        }
    }

    template<typename U, typename BufferMap, typename Field>
    void unpack(const BufferMap& buffer_map, Field& field) const
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<U*>(p.second);
            std::size_t j = 0;
            for (const auto& box : m_outer.at(rank).second)
            {
                // this is a bit cheating
                detail::for_loop_simple<D,D,layout_type>::apply(
                    [buffer,&j,&field](std::size_t offset, std::size_t j_offset)
                    {
                        field.ptr[offset] = buffer[j+j_offset]; 
                    }, 
                    box.first(), 
                    box.last(),
                    field.ext
                );
                j+=box.size();
            }
        }
    }
};


} // namespace ghex

#endif /* INCLUDED_COMMUNICATION_OBJECT_RT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

