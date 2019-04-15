#ifndef INCLUDED_EXCHANGE_HPP
#define INCLUDED_EXCHANGE_HPP

#include "protocol.hpp"
#include "util.hpp"
#include "communication_object.hpp"

#include <mpi.h>

#include <memory>


namespace ghex {


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
        ledger() noexcept
        : total_size{0} { offset.fill(0); num_elements.fill(0); }
        ledger(const ledger&) noexcept = default;
        ledger(ledger&&) noexcept = default;
        std::size_t total_size = 0;
        std::array<std::size_t, size::value> offset;
        std::array<std::size_t, size::value> num_elements;
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
        // compute memory requirements as well as offsets
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
                    l.num_elements[i] = p.second.first;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t);//-1;
                }
                for (const auto& p : co.outer())
                {
                    auto& l         = this->m_outer_ledger_map[p.first];
                    l.offset[i]     = l.total_size;
                    l.num_elements[i] = p.second.first;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t);//-1;
                }
                ++i;
            }
        );

        // allocate the memory per rank
        // and prepare the package 
        for (const auto& p : m_inner_ledger_map)
        {
            m_inner_memory.push_back(memory_holder());
            auto& holder = m_inner_memory.back();
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
                void* ptr_tmp = ptr+x;
                std::size_t space = max_alignment_t::value;
                if (p.second.num_elements[j] > 0)
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
                void* ptr_tmp = ptr+x;
                std::size_t space = max_alignment_t::value;
                if (p.second.num_elements[j] > 0)
                    m_outer_pack[j][p.first] = std::align(m_alignment[j], 1, ptr_tmp, space);
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
    void pack(const Fields&... fields) const
    {
        int i=0;
        detail::for_each(m_cos, std::tuple<const Fields&...>{fields...}, 
            [&i,this](auto& co, auto& field) 
            { 
                co.pack(this->m_inner_pack[i], field);
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
            MPI_Isend(h.aligned_ptr, h.aligned_size, MPI_BYTE, h.rank, 0, m_comm, &m_reqs[i++]);
        }
        // post receives
        for (auto& h : m_outer_memory)
        {
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
    void unpack(Fields&... fields) const
    {
        int i=0;
        detail::for_each(m_cos, std::tuple<Fields&...>{fields...}, 
            [&i,this](auto& co, auto& field) 
            { 
                co.unpack(this->m_outer_pack[i], field);
                ++i;
            }
        );
    }

private:

};


} // namespace ghex


#endif /* INCLUDED_EXCHANGE_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

