#ifndef INCLUDED_COMMUNICATION_OBJECT_HPP
#define INCLUDED_COMMUNICATION_OBJECT_HPP

#include "util.hpp"
#include "regular_domain.hpp"
#include <functional>
#include <mpi.h>
#include <memory>

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
    void pack(BufferMap& buffer_map, const Field& field) /*const*/
    {
        std::cout << "should be packing a map of size " << buffer_map.size() << std::endl;
        std::size_t i = 0;
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            for (const auto& box : m_inner[rank].second)
            {
                std::cout << "  " 
                << "[" <<  box.first()[0] << ", " << box.first()[1] << ", " << box.first()[2] << "] "
                << "[" <<  box.last()[0] << ", " << box.last()[1] << ", " << box.last()[2] << "] "
                << std::endl;
                for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                buffer[i++] = field(x,y,z);
            }
        }
    }

    template<typename BufferMap, typename Field>
    void unpack(const BufferMap& buffer_map, Field& field) const
    {
        std::cout << "should be unpacking a map of size " << buffer_map.size() << std::endl;
        std::size_t i = 0;
        for (auto& p : buffer_map)
        {
            int rank = p.first;
            auto buffer = reinterpret_cast<value_type*>(p.second);
            for (const auto& box : m_outer[rank].second)
            {
                std::cout << "  " 
                << "[" <<  box.first()[0] << ", " << box.first()[1] << ", " << box.first()[2] << "] "
                << "[" <<  box.last()[0] << ", " << box.last()[1] << ", " << box.last()[2] << "] "
                << std::endl;
                for (auto x=box.first()[0]; x<=box.last()[0]; ++x)
                for (auto y=box.first()[1]; y<=box.last()[1]; ++y)
                for (auto z=box.first()[2]; z<=box.last()[2]; ++z)
                field(x,y,z) = buffer[i++];
            }
        }
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

    struct memory_holder
    {
        char* ptr;
        std::size_t size;
        char* aligned_ptr;
        std::size_t aligned_size;
        int rank;
    };

    using ledger_map_type = std::map<int, ledger>;    
    using pack_type = std::array< std::map<int, void*>, size::value>; 

    MPI_Comm                                  m_comm;
    communication_objects_tuple               m_cos;
    allocator_t                               m_alloc;
    std::array<std::size_t, size::value>      m_alignment;
    ledger_map_type                           m_inner_ledger_map;
    ledger_map_type                           m_outer_ledger_map;
    pack_type                                 m_inner_pack;
    pack_type                                 m_outer_pack;
    //std::vector<std::pair<char*,std::size_t>> m_inner_memory;
    //std::vector<std::pair<char*,std::size_t>> m_outer_memory;
    std::vector<memory_holder>                m_inner_memory;
    std::vector<memory_holder>                m_outer_memory;
    std::vector<MPI_Request>                  m_reqs;

public:

    exchange(MPI_Comm comm, communication_object<Domain,T,protocol::mpi_async,Backend>&... cos, const Allocator& alloc = Allocator())
    :   m_comm{comm},
        m_cos{cos...},
        m_alloc{ alloc },
        m_alignment{ alignof(T)... }
    {
        // loop over tuples of communication objects 
        // and compute memory requirements as well as
        // offsets
        int i = 0;
        detail::for_each(
            m_cos, [&i, this](const auto& co) 
            {
                using co_t    = std::remove_reference_t<decltype(co)>;
                using value_t = typename co_t::value_type; 

                for (const auto& p : co.inner())
                {
                    auto& l         = this->m_inner_ledger_map[p.first];
                    l.offset[i]     = l.total_size;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t)-1;
                }
                for (const auto& p : co.outer())
                {
                    auto& l         = this->m_outer_ledger_map[p.first];
                    l.offset[i]     = l.total_size;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t)-1;
                }
                ++i;
            }
        );
        // allocate the memory per rank
        // and prepare the package 
        for (const auto& p : m_inner_ledger_map)
        {
            /*m_inner_memory.push_back(
                std::pair<char*,std::size_t>{
                    reinterpret_cast<char*>(std::allocator_traits<allocator_t>::allocate(m_alloc, p.second.total_size+max_alignment_t::value)),
                    p.second.total_size+max_alignment_t::value
                }
            );*/
            m_inner_memory.push_back(memory_holder());
            auto& holder = m_inner_memory.back();
            holder.ptr = reinterpret_cast<char*>(std::allocator_traits<allocator_t>::allocate(m_alloc, p.second.total_size+max_alignment_t::value));
            holder.size = p.second.total_size+max_alignment_t::value;
            void* buffer =  holder.ptr; //m_inner_memory.back().first;
            std::size_t space = holder.size; // m_inner_memory.back().second;
            char* ptr = reinterpret_cast<char*>(std::align(max_alignment_t::value, 1, buffer, space));
            holder.aligned_ptr = ptr;
            holder.aligned_size = space;
            holder.rank = p.first;
            int j=0;
            for (const auto& x : p.second.offset)
            {
                ptr += x;
                void* ptr_tmp = ptr;
                std::size_t space = max_alignment_t::value;
                m_inner_pack[j][p.first] = std::align(m_alignment[j], 1, ptr_tmp, space);
                ++j;
            }
        }
        for (const auto& p : m_outer_ledger_map)
        {
            m_outer_memory.push_back(memory_holder());
            auto& holder = m_outer_memory.back();
            holder.ptr = reinterpret_cast<char*>(std::allocator_traits<allocator_t>::allocate(m_alloc, p.second.total_size+max_alignment_t::value));
            holder.size = p.second.total_size+max_alignment_t::value;
            void* buffer =  holder.ptr;
            std::size_t space = holder.size;
            char* ptr = reinterpret_cast<char*>(std::align(max_alignment_t::value, 1, buffer, space));
            holder.aligned_ptr = ptr;
            holder.aligned_size = space;
            holder.rank = p.first;
            int j=0;
            for (const auto& x : p.second.offset)
            {
                ptr += x;
                void* ptr_tmp = ptr;
                std::size_t space = max_alignment_t::value;
                m_inner_pack[j][p.first] = std::align(m_alignment[j], 1, ptr_tmp, space);
                ++j;
            }
        }
        m_reqs.resize(m_inner_memory.size()+m_outer_memory.size());
    }

    ~exchange()
    {
        for (auto& h : m_inner_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, h.ptr, h.size);
        for (auto& h : m_outer_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, h.ptr, h.size);
    }

    template<typename... Fields>
    void pack(Fields&... fields)
    {
        int i=0;
        detail::for_each(m_cos, std::tuple<Fields&...>{fields...}, 
            [&i,this](auto& co, auto& field) 
            { 
                co.pack( this->m_inner_pack[i], field );
                ++i;
            }
        );
    }

    void post()
    {
        // post sends
        int i = 0;
        for (auto& h : m_inner_memory)
        {
            std::cout << "recv from " << h.rank << " " << h.aligned_size << " bytes" << std::endl;
            MPI_Isend(h.aligned_ptr, h.aligned_size, MPI_BYTE, h.rank, 0, m_comm, &m_reqs[i++]);
        }
        // post receives
        for (auto& h : m_outer_memory)
        {
            std::cout << "send to   " << h.rank << " " << h.aligned_size << " bytes" << std::endl;
            MPI_Irecv(h.aligned_ptr, h.aligned_size, MPI_BYTE, h.rank, 0, m_comm, &m_reqs[i++]);
        }
    }

    void wait()
    {
        // wait for exchange to finish
        std::vector<MPI_Status> sts(m_reqs.size());
        MPI_Waitall(m_reqs.size(), &m_reqs[0], &sts[0]);
    }

    template<typename... Fields>
    void unpack(Fields&... fields)
    {
        int i=0;
        detail::for_each(m_cos, std::tuple<Fields&...>{fields...}, 
            [&i,this](auto& co, auto& field) 
            { 
                co.pack( this->m_outer_pack[i], field );
                ++i;
            }
        );
    }

private:

};

} // namespace ghex


#endif /* INCLUDED_COMMUNICATION_OBJECT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

