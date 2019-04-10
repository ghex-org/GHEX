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

                //std::cout << std::endl;
                //std::cout << "co.inner().size = " << co.inner().size() << std::endl;
                for (const auto& p : co.inner())
                {
                    auto& l         = this->m_inner_ledger_map[p.first];
                    //std::cout << "total size before: " << l.total_size << std::endl;
                    l.offset[i]     = l.total_size;
                    //std::cout << "adding " << p.second.first << " elements" << std::endl;
                    //std::cout << "sizeof(value_t) = " << sizeof(value_t) << std::endl;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t);//-1;
                    //std::cout << "bytes = " << p.second.first*sizeof(value_t) + alignof(value_t)/*-1*/ << std::endl;
                    //std::cout << "total size after: " << l.total_size << std::endl;
                }
                for (const auto& p : co.outer())
                {
                    auto& l         = this->m_outer_ledger_map[p.first];
                    l.offset[i]     = l.total_size;
                    l.total_size   += p.second.first*sizeof(value_t) + alignof(value_t);//-1;
                }
                ++i;
            }
        );

        //std::cout << std::endl;

        // allocate the memory per rank
        // and prepare the package 
        for (const auto& p : m_inner_ledger_map)
        {
            m_inner_memory.push_back(memory_holder());
            auto& holder = m_inner_memory.back();
            holder.ptr = reinterpret_cast<char*>(std::allocator_traits<allocator_t>::allocate(m_alloc, p.second.total_size+max_alignment_t::value));
            //std::cout << "total size is now " << p.second.total_size << " bytes" << std::endl;
            //holder.ptr = ::new char[p.second.total_size+max_alignment_t::value /*+ 1000*/];
            //std::cout << "allocating " << p.second.total_size+max_alignment_t::value << " bytes for inner at rank " << p.first << std::endl;
            holder.size = p.second.total_size+max_alignment_t::value;

            //std::cout << "total size should be: " << 26*12+4+(26*24+8)*2 << std::endl;
            //std::cout << "ptrs : " << (void*)(holder.ptr) 
            //    << " " << (void*)(holder.ptr+26*12+4)
            //    << " " << (void*)(holder.ptr+26*12+4+26*24+8)
            //    << " " << (void*)(holder.ptr+26*12+4+(26*24+8)*2+max_alignment_t::value)
            //    << std::endl;

            void* buffer =  holder.ptr; //m_inner_memory.back().first;
            std::size_t space = holder.size; // m_inner_memory.back().second;
            char* ptr = reinterpret_cast<char*>(std::align(max_alignment_t::value, 1, buffer, space));
            holder.aligned_ptr = ptr;
            holder.aligned_size = space;
            holder.rank = p.first;
            //std::cout << (void*)(holder.aligned_ptr) << " " << (void*)(holder.aligned_ptr+holder.aligned_size) << std::endl;
            //std::cout << (void*)(holder.ptr) << " " << (void*)(holder.ptr+holder.size) << std::endl;
            int j=0;
            for (const auto& x : p.second.offset)
            {
                //char* location = ptr+x;
                //std::cout << "offset = " << x << std::endl;
                //ptr += x;
                void* ptr_tmp = ptr+x;
                std::size_t space = max_alignment_t::value;
                //std::cout << "alignment " << j << " = " << m_alignment[j] << std::endl;
                //std::cout << "ptr before alignment: " << ptr_tmp << std::endl;
                m_inner_pack[j][p.first] = std::align(m_alignment[j], 1, ptr_tmp, space);
                //std::cout << "ptr after alignment: " << m_inner_pack[j][p.first] << std::endl;
                //std::cout << "ptr after alignment end: " << reinterpret_cast<void*>(reinterpret_cast<char*>(m_inner_pack[j][p.first]) + 26*24) << std::endl; 
                //std::cout << "ptr after alignment end: " << 26*24 << " bytes" << std::endl; 
                ++j;
            }
        }
        //std::cout << m_inner_pack[0][0] << std::endl;
        //std::cout << m_inner_pack[1][0] << std::endl;
        //std::cout << m_inner_pack[2][0] << std::endl;
        for (const auto& p : m_outer_ledger_map)
        {
            m_outer_memory.push_back(memory_holder());
            auto& holder = m_outer_memory.back();
            holder.ptr = reinterpret_cast<char*>(std::allocator_traits<allocator_t>::allocate(m_alloc, p.second.total_size+max_alignment_t::value));
            //holder.ptr = ::new char[p.second.total_size+max_alignment_t::value+ 1000];
            //std::cout << "allocating " << p.second.total_size+max_alignment_t::value << " bytes for outer at rank " << p.first << std::endl;
            holder.size = p.second.total_size+max_alignment_t::value;
            void* buffer =  holder.ptr;
            std::size_t space = holder.size;
            char* ptr = reinterpret_cast<char*>(std::align(max_alignment_t::value, 1, buffer, space));
            holder.aligned_ptr = ptr;
            holder.aligned_size = space;
            holder.rank = p.first;
            //std::cout << "holder space: " << holder.aligned_size << std::endl;
            //std::cout << (void*)(holder.aligned_ptr) << " " << (void*)(holder.aligned_ptr+holder.aligned_size) << std::endl;
            int j=0;
            for (const auto& x : p.second.offset)
            {
                //char* location = ptr+x;
                void* ptr_tmp = ptr+x;
                std::size_t space = max_alignment_t::value;
                m_outer_pack[j][p.first] = std::align(m_alignment[j], 1, ptr_tmp, space);
                ++j;
            }
        }

        //std::cout << m_outer_pack[0][0] << std::endl;
        //std::cout << m_outer_pack[1][0] << std::endl;
        //std::cout << m_outer_pack[2][0] << std::endl;
        m_reqs.resize(m_inner_memory.size()+m_outer_memory.size());
    }

    ~exchange()
    {
        for (auto& h : m_inner_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, h.ptr, h.size);
            //delete[] h.ptr;
        for (auto& h : m_outer_memory)
            std::allocator_traits<allocator_t>::deallocate(m_alloc, h.ptr, h.size);
            //delete[] h.ptr;
    }

    template<typename... Fields>
    void pack(const Fields&... fields) const
    {
        int i=0;
        detail::for_each(m_cos, std::tuple<const Fields&...>{fields...}, 
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
            //std::cout << "recv from " << h.rank << " " << h.aligned_size << " bytes" << std::endl;
            MPI_Isend(h.aligned_ptr, h.aligned_size, MPI_BYTE, h.rank, 0, m_comm, &m_reqs[i++]);
        }
        // post receives
        for (auto& h : m_outer_memory)
        {
            //std::cout << "send to   " << h.rank << " " << h.aligned_size << " bytes" << std::endl;
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
                co.unpack( this->m_outer_pack[i], field );
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

