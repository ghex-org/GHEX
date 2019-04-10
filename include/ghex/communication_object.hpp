#ifndef INCLUDED_COMMUNICATION_OBJECT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_HPP

#include "util.hpp"
#include "regular_domain.hpp"
#include "protocol.hpp"
//#include <functional>
//#include <mpi.h>
//#include <memory>

namespace ghex {


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
    void pack(BufferMap& buffer_map, const Field& field) /*const*/
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            std::size_t i = 0;
            for (const auto& box : m_inner[rank].second)
            {
                /*std::cout << "  " 
                << "[" <<  box.first()[0] << ", " << box.first()[1] << ", " << box.first()[2] << "] "
                << "[" <<  box.last()[0] << ", " << box.last()[1] << ", " << box.last()[2] << "] "
                << std::endl;*/
                for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                //{std::cout << "  " << (void*)(buffer+i) << " i= " << i << std::endl;
                //    std::cout << (reinterpret_cast<char*>(buffer+i+1) - reinterpret_cast<char*>(buffer+i)) << std::endl;
                    buffer[i++] = field(x,y,z);
            }
        }
    }

    template<typename BufferMap, typename Field>
    void unpack(/*const*/ BufferMap& buffer_map, Field& field) /*const*/
    {
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            std::size_t i = 0;
            for (const auto& box : m_outer[rank].second)
            {
                /*std::cout << "  " 
                << "[" <<  box.first()[0] << ", " << box.first()[1] << ", " << box.first()[2] << "] "
                << "[" <<  box.last()[0] << ", " << box.last()[1] << ", " << box.last()[2] << "] "
                << std::endl;*/
                for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                field(x,y,z) = buffer[i++];
            }
        }
    }
};



} // namespace ghex


#endif /* INCLUDED_COMMUNICATION_OBJECT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

