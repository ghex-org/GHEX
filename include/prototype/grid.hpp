#ifndef INCLUDED_GRID_HPP
#define INCLUDED_GRID_HPP


#include <iostream>
#include <map>
#include <vector>
#include <algorithm>
#include <mpi.h>

namespace gridtools {


template<
    typename LocalCellId,
    typename GlobalCellId,
    typename DomainId,
    typename Extent,
    typename IdMap
>
struct regular_grid_map
{

    using local_cell_id_t = LocalCellId;
    using global_cell_id_t = GlobalCellId;
    using domain_id_t = DomainId;
    using extent_t = Extent;
    using id_map_t = IdMap;

    struct range
    {
        local_cell_id_t m_begin;
        local_cell_id_t m_end;
        global_cell_id_t m_first;
        global_cell_id_t m_last;

        bool operator<(const range& other) const
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

    extent_t m_halo;
    extent_t m_k;
    local_cell_id_t m_begin;
    local_cell_id_t m_end;
    id_map_t m_id_map;

    std::map<domain_id_t, std::vector<range> > m_send_ranges;
    std::map<domain_id_t, std::vector<range> > m_recv_ranges;

    regular_grid_map(extent_t halo, extent_t k, local_cell_id_t b, local_cell_id_t e, id_map_t m) :
        m_halo(halo),
        m_k(k),
        m_begin(b),
        m_end(e),
        m_id_map(m)
    {

        // generate recv ranges 
        // here we know domain is 2D and halo is symmetric
        // but can be easily made more general
        // Assumption: local_cell_id_t is a coordinate
        
        {   // southwest
            local_cell_id_t origin(m_begin);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // south
            local_cell_id_t origin(m_begin[0]+halo,m_begin[1]);
            local_cell_id_t end(origin[0]+(m_end[0]-m_begin[0]-2*halo),origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // southeast
            local_cell_id_t origin(m_begin[0]+(m_end[0]-m_begin[0]-halo),m_begin[1]);
            local_cell_id_t end(m_end[0],origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }

        {   // west
            local_cell_id_t origin(m_begin[0],m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+(m_end[1]-m_begin[1]-2*halo));
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // east
            local_cell_id_t origin(m_begin[0]+(m_end[0]-m_begin[0]-halo),m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+(m_end[1]-m_begin[1]-2*halo));
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }


        {   // northwest
            local_cell_id_t origin(m_begin[0],m_end[1]-halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // north
            local_cell_id_t origin(m_begin[0]+halo,m_end[1]-halo);
            local_cell_id_t end(origin[0]+(m_end[0]-m_begin[0]-2*halo),origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // northeast
            local_cell_id_t origin(m_begin[0]+(m_end[0]-m_begin[0]-halo),m_end[1]-halo);
            local_cell_id_t end(m_end[0],origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            m_recv_ranges[std::get<1>(m_id_map(origin))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }

        for (auto& p : m_recv_ranges)
            std::sort(p.second.begin(), p.second.end());

        // generate send ranges 
        // here we know domain is 2D and halo is symmetric
        // but can be easily made more general
        // Assumption: local_cell_id_t is a coordinate
        
        {   // southwest
            local_cell_id_t origin(m_begin[0]+halo,m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0]-1,origin[1]-1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // south
            local_cell_id_t origin(m_begin[0]+halo,m_begin[1]+halo);
            local_cell_id_t end(origin[0]+(m_end[0]-m_begin[0]-2*halo),origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0],origin[1]-1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // southeast
            local_cell_id_t origin(m_end[0]-2*halo,m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0]+1,origin[1]-1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }

        {   // west
            local_cell_id_t origin(m_begin[0]+halo,m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+(m_end[1]-m_begin[1]-2*halo));
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0]-1,origin[1]);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // east
            local_cell_id_t origin(m_end[0]-2*halo,m_begin[1]+halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+(m_end[1]-m_begin[1]-2*halo));
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(last[0]+1,origin[1]);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }

        {   // northwest
            local_cell_id_t origin(m_begin[0]+halo,m_end[1]-2*halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0]-1,last[1]+1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // north
            local_cell_id_t origin(m_begin[0]+halo,m_end[1]-2*halo);
            local_cell_id_t end(origin[0]+(m_end[0]-m_begin[0]-2*halo),origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(origin[0],last[1]+1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }
        {   // northeast
            local_cell_id_t origin(m_end[0]-2*halo,m_end[1]-2*halo);
            local_cell_id_t end(origin[0]+halo,origin[1]+halo);
            local_cell_id_t last(end[0]-1,end[1]-1);
            local_cell_id_t target(last[0]+1,last[1]+1);
            m_send_ranges[std::get<1>(m_id_map(target))].push_back(
                range{
                    origin, 
                    end, 
                    std::get<0>(m_id_map(origin)),   
                    std::get<0>(m_id_map(last))
                }
            );
        }

        for (auto& p : m_send_ranges)
            std::sort(p.second.begin(), p.second.end());
    }

    template<typename Container, typename Grid>
    void pack(Container& container, const Grid& g)
    {
        container.resize(m_send_ranges.size());
        int q = 0;
        for (const auto& p : m_send_ranges)
        {
            const auto& ranges = p.second;
            int size = 0;
            for (const auto& r : ranges)
                size += (r.m_end[0]-r.m_begin[0])*(r.m_end[1]-r.m_begin[1]);
            size *= m_k;

            container[q].resize(size);

            int i = 0;
            for (const auto& r : ranges)
            {
                for (int y=r.m_begin[1]; y<r.m_end[1]; ++y)
                for (int x=r.m_begin[0]; x<r.m_end[0]; ++x)
                {
                    local_cell_id_t loc_id(x,y);
                    for (int z=0; z<m_k; ++z)
                    {
                        container[q][i++] = g(loc_id,z);  
                    }
                }
            }
            ++q;
        }
    }

    template<typename Container, typename Grid>
    void unpack(const Container& container, Grid& g)
    {
        int q = 0;
        for (const auto& p : m_recv_ranges)
        {
            const auto& ranges = p.second;

            int i = 0;
            for (const auto& r : ranges)
            {
                for (int y=r.m_begin[1]; y<r.m_end[1]; ++y)
                for (int x=r.m_begin[0]; x<r.m_end[0]; ++x)
                {
                    local_cell_id_t loc_id(x,y);
                    for (int z=0; z<m_k; ++z)
                    {
                        g(loc_id,z) = container[q][i++];  
                    }
                }
            }
            ++q;
        }
    }

    template<typename Container, typename F>
    std::vector<MPI_Request> exchange(const Container& send_container, Container& recv_container, F&& get_rank)
    {
        recv_container.resize(m_recv_ranges.size());
        int q = 0;
        for (const auto& p : m_recv_ranges)
        {
            const auto& ranges = p.second;
            int size = 0;
            for (const auto& r : ranges)
                size += (r.m_end[0]-r.m_begin[0])*(r.m_end[1]-r.m_begin[1]);
            size *= m_k;

            recv_container[q++].resize(size);
        }


        std::vector<MPI_Request> reqs(send_container.size()+recv_container.size());

        const auto my_domain = std::get<1>(m_id_map(local_cell_id_t(m_begin[0]+m_halo,m_begin[1]+m_halo)));
        const int my_rank = get_rank(std::get<1>(m_id_map(m_begin)));

        // post receives
        int r = 0;
        int mr = 0;
        for (const auto& p : m_recv_ranges)
        {
            int src_rank = get_rank(p.first);
            int tag = my_domain;
            char* buffer = const_cast<char*>(reinterpret_cast<const char*>(&recv_container[mr][0]));
            const unsigned int size = const_cast<char*>(reinterpret_cast<const char*>(&recv_container[mr].back() + 1)) - buffer;

            MPI_Irecv(
                buffer, 
                size, 
                MPI_BYTE, 
                src_rank, 
                tag, 
                MPI_COMM_WORLD, 
                &reqs[r++]
            );
            ++mr;
        }

        // post sends
        int ms = 0;
        for (const auto& p : m_send_ranges)
        {
            int dest_rank = get_rank(p.first);
            int tag = p.first;
            char* buffer = const_cast<char*>(reinterpret_cast<const char*>(&send_container[ms][0]));
            const unsigned int size = const_cast<char*>(reinterpret_cast<const char*>(&send_container[ms].back() + 1)) - buffer;

            MPI_Isend(
                buffer, 
                size, 
                MPI_BYTE, 
                dest_rank, 
                tag, 
                MPI_COMM_WORLD, 
                &reqs[r++]
            );
            ++ms;
        }

        return std::move(reqs);
    }


};


} // namespace gridtools


#endif /* INCLUDED_GRID_HPP */

