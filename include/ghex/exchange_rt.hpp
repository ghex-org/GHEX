#ifndef INCLUDED_EXCHANGE_RT_HPP
#define INCLUDED_EXCHANGE_RT_HPP

#include <boost/align/aligned_allocator_adaptor.hpp>
#include <functional>
#include <set>
#include <iostream>

namespace ghex {

template<typename Allocator = std::allocator<char>>
class exchange_rt
{
public:

    using allocator_t = typename std::allocator_traits<Allocator>::template rebind_alloc<char>;
    // make sure this allocator uses big enough fixed alignment!
    //using alligned_allocator_t = boost::alignment::aligned_allocator_adaptor<allocator_t, 64>;
    using aligned_allocator_t = boost::alignment::aligned_allocator_adaptor<allocator_t, alignof(double)>;

    exchange_rt(MPI_Comm comm, const Allocator& alloc = Allocator()) :
        m_comm{comm},
        m_alloc{alloc},
        m_aligned_alloc{m_alloc}
    {}


    template<typename CommunicationObject, typename Field>
    void pack(const CommunicationObject& co, const Field& f)
    {
        using value_t = std::remove_cv_t<typename Field::value_type>;

        std::map<int, void*> inner_pack;

        m_offset.resize(m_offset.size()+1);
        m_alignments.push_back(alignof(value_t));

        for (const auto& p: co.inner())
        {
            auto it = m_inner_mem.find(p.first);
            if (it == m_inner_mem.end()) 
                it = m_inner_mem.insert( std::make_pair(p.first, memory_vec_t{m_aligned_alloc} ) ).first;
            auto& mem_vec = it->second;
            const auto prev_size = mem_vec.size();
            //std::cout << "resizing inner pack " << p.first << " from " << prev_size << " to " << prev_size+p.second.first*sizeof(value_t)+alignof(value_t) << std::endl;
            //std::cout << "increasing inner pack " << p.first <<  " with " << p.second.first << std::endl;
            mem_vec.resize(mem_vec.size()+p.second.first*sizeof(value_t)+alignof(value_t));
            char* ptr = &mem_vec[prev_size];
            void* ptr_tmp = ptr;
            std::size_t space = p.second.first*sizeof(value_t)+alignof(value_t);
            if (p.second.first > 0)
            {
                inner_pack[p.first] = std::align(alignof(value_t), 1, ptr_tmp, space);
                m_inner_ranks.insert(p.first);
            }
        }
        for (const auto& p: co.outer())
        {
            auto it = m_outer_size.find(p.first);
            if (it == m_outer_size.end())
                it = m_outer_size.insert( std::make_pair(p.first, std::size_t(0)) ).first;
            auto& s = it->second;
            const auto prev_size = s;
            s += p.second.first*sizeof(value_t)+alignof(value_t);
            if (p.second.first > 0)
            {
                m_offset.back()[p.first] = prev_size;
                m_outer_ranks.insert(p.first);
            }
        }

        func_vec.push_back( 
            [&f,&co](const std::map<int,void*>& m )
            {
                co.template unpack<value_t>(m, const_cast<Field&>(f) );
            }
        );

        co.template pack<value_t>(inner_pack, f);
    }

    void post()
    {
        for (auto& p : m_outer_size)
        {
            auto it = m_outer_mem.find(p.first);
            if (it == m_outer_mem.end()) 
                it = m_outer_mem.insert( std::make_pair(p.first, memory_vec_t{m_aligned_alloc} ) ).first;
            auto& mem_vec = it->second;
            mem_vec.resize(p.second);
            p.second = 0;
        }
        m_outer_packs.resize(m_offset.size());
        for (unsigned int i=0; i<m_offset.size(); ++i)
        {
            const auto& offset_map = m_offset[i];
            auto& outer_pack = m_outer_packs[i];
            for (const auto& p : offset_map)
            {
                const auto rank = p.first;
                const auto offset = p.second;
                char* ptr = &m_outer_mem[rank][offset];
                void* ptr_tmp = ptr;
                std::size_t space = m_alignments[i];
                outer_pack[rank] = std::align(m_alignments[i], 1, ptr_tmp, space);
            }

        }

        m_reqs.resize(m_outer_ranks.size()+m_inner_ranks.size());

        int i=0;
        for (const auto& r : m_inner_ranks)
        {
            auto& buffer = m_inner_mem[r];
            MPI_Isend(&buffer[0], buffer.size(), MPI_BYTE, r, 0, m_comm, &m_reqs[i++]);
        }

        for (const auto& r : m_outer_ranks)
        {
            auto& buffer = m_outer_mem[r];
            MPI_Irecv(&buffer[0], buffer.size(), MPI_BYTE, r, 0, m_comm, &m_reqs[i++]);
        }


    }

    void wait()
    {
        // wait for exchange to finish
        std::vector<MPI_Status> sts(m_reqs.size());
        MPI_Waitall(m_reqs.size(), &m_reqs[0], &sts[0]);
        for (auto& p : m_inner_mem) p.second.resize(0);
    }


    void unpack()
    {
        for (unsigned int i=0; i<func_vec.size(); ++i)
        {
            func_vec[i](m_outer_packs[i]);
        }

        for (auto& p : m_outer_mem) p.second.resize(0);
        m_offset.resize(0);
        func_vec.resize(0);
        m_alignments.resize(0);
        m_outer_packs.resize(0);
        m_outer_ranks.clear();
        m_inner_ranks.clear();
    }


    void clear()
    {
        for (auto& p : m_outer_size) p.second = 0;
        for (auto& p : m_inner_mem) p.second.resize(0);
        for (auto& p : m_outer_mem) p.second.resize(0);
    }

private:

    using memory_vec_t = std::vector<char, aligned_allocator_t>;
    using map_t = std::map<int, memory_vec_t>;
    using offset_t = std::vector<std::map<int,std::size_t>>;
    using func_vec_t = std::vector<std::function<void(const std::map<int,void*>&)>>;

    MPI_Comm m_comm;
    allocator_t m_alloc;
    aligned_allocator_t m_aligned_alloc;
    std::map<int, std::size_t> m_outer_size;
    map_t m_inner_mem;
    map_t m_outer_mem;
    offset_t m_offset;
    func_vec_t func_vec;
    std::vector<std::size_t> m_alignments;
    std::vector<std::map<int,void*>> m_outer_packs;
    std::set<int> m_inner_ranks;
    std::set<int> m_outer_ranks;
    std::vector<MPI_Request>                  m_reqs;
};



} // namespace ghex


#endif /* INCLUDED_EXCHANGE_RT_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

