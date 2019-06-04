/* 
 * GridTools
 * 
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_STRUCTURED_PATTERN_HPP
#define INCLUDED_STRUCTURED_PATTERN_HPP

#include "protocol/communicator_base.hpp"
#include "pattern.hpp"
#include "coordinate.hpp"
#include <map>

namespace gridtools {

    namespace detail {

        template<typename CoordinateArrayType>
        struct structured_grid 
        {
            using coordinate_base_type    = CoordinateArrayType;
            using coordinate_type         = coordinate<coordinate_base_type>;
            using coordinate_element_type = typename coordinate_type::element_type;
            using dimension               = typename coordinate_type::dimension;    
        };

    } // namespace detail

    /** @brief type to indicate structured grids */
    struct structured_grid 
    {
        template<typename Domain>
        using type = detail::structured_grid<typename Domain::coordinate_type>;
    };

    /** @brief structured pattern */
    template<typename P, typename CoordinateArrayType, typename DomainIdType>
    class pattern<P,detail::structured_grid<CoordinateArrayType>,DomainIdType>
    {
    public: // member types

        using grid_type               = detail::structured_grid<CoordinateArrayType>;
        using coordinate_type         = typename grid_type::coordinate_type;
        using coordinate_element_type = typename grid_type::coordinate_element_type;
        using dimension               = typename grid_type::dimension;
        using communicator_type       = protocol::communicator<P>;
        using address_type            = typename communicator_type::address_type;
        using domain_id_type          = DomainIdType;

        struct iteration_space
        {
                  coordinate_type& first()       noexcept { return _min; }
                  coordinate_type& last()        noexcept { return _max; }
            const coordinate_type& first() const noexcept { return _min; }
            const coordinate_type& last()  const noexcept { return _max; }

            iteration_space intersect(iteration_space x, bool& found) const noexcept
            {
                x.first() = max(x.first(), first());
                x.last()  = min(x.last(),  last());
                found = (x.first <= x.last());
                return std::move(x);
            }

            int size() const noexcept 
            {
                int s = _max[0]-_min[0]+1;
                for (int i=1; i<coordinate_type::size(); ++i) s *= _max[i]-_min[i]+1;
                return s;
            }

            coordinate_type _min; 
            coordinate_type _max;
        };

        struct extended_domain_id_type
        {
            domain_id_type id;
            int            mpi_rank;
            address_type   address;
            int            tag;

            bool operator<(const extended_domain_id_type& other) const noexcept { return id < other.id; }
        };

        using index_container_type = std::vector<iteration_space>;
        using map_type = std::map<extended_domain_id_type, index_container_type>;

        static int num_elements(const index_container_type& c) noexcept
        {
            int s = 0;
            for (const auto& is : c) s += is.size();
            return s;
        }

    private: // members

        communicator_type m_comm;
        iteration_space m_domain;
        extended_domain_id_type m_id;
        map_type m_send_map;
        map_type m_recv_map;

    public: // ctors

        pattern(communicator_type& comm, const iteration_space& domain, const extended_domain_id_type& id)
        :   m_comm(comm), m_domain(domain), m_id(id)
        {}

        pattern(const pattern&) = default;

        pattern(pattern&&) = default;

    public: // member functions

        map_type& send_halos() noexcept { return m_send_map; }
        const map_type& send_halos() const noexcept { return m_send_map; }

        map_type& recv_halos() noexcept { return m_recv_map; }
        const map_type& recv_halos() const noexcept { return m_recv_map; }

        domain_id_type domain_id() const noexcept { return m_id.id; }
        extended_domain_id_type extended_domain_id() const noexcept { return m_id; }

        /*template<typename Device, typename Field>
        buffer_info<pattern, Device, Field> bind(Field& field, typename Device::id_type id) const
        {
            return {*this,field,id};
        }*/
        template<typename Field>
        buffer_info<pattern, typename Field::device_type, Field> operator()(Field& field) const
        {
            return {*this,field,field.device_id()};
        }
    };

    namespace detail {

        template<typename CoordinateArrayType>
        struct make_pattern_impl<detail::structured_grid<CoordinateArrayType>>
        {
            template<typename P, typename HaloGenerator, typename DomainRange>
            static auto apply(protocol::setup_communicator& comm, protocol::communicator<P>& new_comm, HaloGenerator&& hgen, DomainRange&& d_range)
            {
                using domain_type               = typename std::remove_reference_t<DomainRange>::value_type;
                using domain_id_type            = typename domain_type::domain_id_type;
                using grid_type                 = detail::structured_grid<CoordinateArrayType>;
                using pattern_type              = pattern<P, grid_type, domain_id_type>;
                using iteration_space           = typename pattern_type::iteration_space;
                using coordinate_type           = typename pattern_type::coordinate_type;
                using extended_domain_id_type   = typename pattern_type::extended_domain_id_type;

                // get this address from new communicator
                auto my_address = new_comm.address();
                
                // set up domain ids, extents and recv halos
                std::vector<iteration_space>               my_domain_extents;
                std::vector<extended_domain_id_type>       my_domain_ids;
                std::vector<pattern_type>                  my_patterns;
                std::vector<std::vector<iteration_space>>  my_generated_recv_halos;
                for (const auto& d : d_range)
                {
                    my_domain_ids.push_back( extended_domain_id_type{d.domain_id(), comm.rank(), my_address, 0} );
                    my_domain_extents.push_back( iteration_space{coordinate_type{d.first()}, coordinate_type{d.last()}} );
                    my_patterns.emplace_back( new_comm, my_domain_extents.back(), my_domain_ids.back() );
                    my_generated_recv_halos.resize(my_generated_recv_halos.size()+1);
                    // generate recv halos
                    auto recv_halos = hgen(d);
                    // convert ghe recv halos to internal format
                    for (const auto& h : recv_halos)
                        my_generated_recv_halos.back().push_back( iteration_space{coordinate_type{h.first()}, coordinate_type{h.last()}} );
                }

                // find all domains and their extents
                int my_num_domains   = my_domain_ids.size();
                auto num_domain_ids  = comm.all_gather(my_num_domains).get();
                auto domain_ids      = comm.all_gather(my_domain_ids, num_domain_ids).get();
                auto domain_extents  = comm.all_gather(my_domain_extents, num_domain_ids).get();
                const int world_size = num_domain_ids.size();
                
                // loop over patterns/domains
                for (unsigned int i=0; i<my_patterns.size(); ++i)
                {
                    // get corresponding halos
                    const auto recv_halos = my_generated_recv_halos[i];
                    // intersect each halo with all domain extents
                    for (const auto& halo : recv_halos)
                    {
                        for (unsigned int j=0; j<domain_extents.size(); ++j)
                        {
                            const auto& extents_vec   = domain_extents[j];
                            const auto& domain_id_vec = domain_ids[j];
                            for (unsigned int k=0; k<extents_vec.size(); ++k)
                            {
                                const auto& extent = extents_vec[k];
                                const auto& domain_id = domain_id_vec[k];

                                const auto left  = max(halo.first(),extent.first());
                                const auto right = min(halo.last(),extent.last());

                                if (left <= right)
                                {
                                    iteration_space h{left, right};
                                    my_patterns[i].recv_halos()[domain_id].push_back(h);
                                }
                            }
                        }
                    }
                }

                // set tags in order to disambiguate recvs from same processor but different domain
                std::map<int,int> tag_map;
                for (auto& p : my_patterns)
                {
                    for (auto& id_is_pair : p.recv_halos())
                    {
                        const int rank = id_is_pair.first.mpi_rank;
                        auto it = tag_map.find(rank);
                        if (it == tag_map.end())
                        {
                            tag_map[rank] = 0;
                            const_cast<extended_domain_id_type&>(id_is_pair.first).tag = 0;
                        }
                        else
                        {
                            ++it->second;
                            const_cast<extended_domain_id_type&>(id_is_pair.first).tag = it->second;
                        }
                    }
                }
                
                // translate to send halos
                std::map<int,
                    std::map<domain_id_type,
                        std::map<extended_domain_id_type, std::vector<iteration_space> > > > send_halos_map;
                for (const auto& p : my_patterns)
                {
                    for (const auto& id_is_pair : p.recv_halos())
                    {
                        auto d_id = p.extended_domain_id();
                        d_id.tag = id_is_pair.first.tag;
                        auto& is_vec = send_halos_map[id_is_pair.first.mpi_rank][id_is_pair.first.id][d_id];
                        is_vec.insert(is_vec.end(), id_is_pair.second.begin(), id_is_pair.second.end());
                    }
                }

                // filter out my own send halos
                auto it = send_halos_map.find(comm.rank());
                if (it != send_halos_map.end())
                {
                    for (const auto& p1 : it->second)
                    {
                        const auto dom_id = p1.first;
                        unsigned int j = 0;
                        for (const auto& p : my_patterns)
                        {
                            if (p.domain_id() == dom_id) break;
                            ++j;
                        }
                        pattern_type& p = my_patterns[j];
                        for (const auto& p2 : p1.second)
                            p.send_halos().insert(p2);
                    }
                    send_halos_map.erase(it);
                }

                // loop over all ranks and establish connection
                for (int rank = 0; rank<world_size; ++rank)
                {
                    if (rank == comm.rank())
                    {
                        // broadcast number of connecting ranks
                        int num_ranks = send_halos_map.size();
                        comm.broadcast(num_ranks, rank);

                        if (num_ranks > 0)
                        {
                            // broadcast ranks
                            std::vector<int> ranks;
                            ranks.reserve(num_ranks);
                            for (const auto& p : send_halos_map) 
                                ranks.push_back(p.first);
                            comm.broadcast(&ranks[0],num_ranks,rank);

                            // send number of domains to each rank
                            std::vector<int> num_domains;
                            num_domains.reserve(num_ranks);
                            for (const auto& p : send_halos_map)
                                num_domains.push_back(p.second.size());
                            int j=0;
                            for (auto& nd : num_domains)
                                comm.send(ranks[j++], 0, nd);

                            j=0;
                            for (const auto& p : send_halos_map)
                            {
                                // send domain ids
                                std::vector<domain_id_type> dom_ids;
                                dom_ids.reserve(num_domains[j]);
                                for (const auto& p1 : p.second)
                                    dom_ids.push_back(p1.first);
                                comm.send(ranks[j], 0, &dom_ids[0], num_domains[j]);

                                // send number of id_iteration_space pairs per domain
                                std::vector<int> num_pairs;
                                num_pairs.reserve(num_domains[j]);
                                for (const auto& p1 : p.second)
                                    num_pairs.push_back(p1.second.size());
                                comm.send(ranks[j], 0, &num_pairs[0], num_domains[j]);

                                int k=0;
                                for (const auto& p1 : p.second)
                                {
                                    // send extended_domain_ids for each domain j and each pair k
                                    std::vector<extended_domain_id_type> my_dom_ids;
                                    my_dom_ids.reserve(num_pairs[k]);
                                    for (const auto& p2 : p1.second)
                                        my_dom_ids.push_back(p2.first);
                                    comm.send(ranks[j], 0, &my_dom_ids[0], num_pairs[k]);

                                    // send all iteration spaces for each domain j, each pair k
                                    for (const auto& p2 :  p1.second)
                                    {
                                        int num_is = p2.second.size();
                                        comm.send(ranks[j], 0, num_is);
                                        comm.send(ranks[j], 0, &p2.second[0], num_is);
                                    }
                                    ++k;
                                }
                                ++j;
                            }
                        } 
                    }
                    else
                    {
                        // broadcast number of connecting ranks
                        int num_ranks;
                        comm.broadcast(num_ranks, rank);

                        if (num_ranks > 0)
                        {
                            // broadcast ranks
                            std::vector<int> ranks(num_ranks);
                            comm.broadcast(&ranks[0],num_ranks,rank);

                            // check if I am part of the ranks
                            bool sending = false;
                            for (auto r : ranks)
                            {
                                if (r == comm.rank())
                                {
                                    sending = true;
                                    break;
                                }
                            }
                            if (sending)
                            {
                                // recv number of domains
                                int num_domains;
                                comm.recv(rank, 0, num_domains);

                                // recv domain ids
                                std::vector<domain_id_type> dom_ids(num_domains);
                                comm.recv(rank, 0, &dom_ids[0], num_domains);

                                // recv number of id_iteration_space pairs per domain
                                std::vector<int> num_pairs(num_domains);
                                comm.recv(rank, 0, &num_pairs[0], num_domains);

                                int j=0;
                                for (auto np : num_pairs)
                                {
                                    // recv extended_domain_ids for each domain j and all its np pairs
                                    std::vector<extended_domain_id_type> send_dom_ids(np);
                                    comm.recv(rank, 0, &send_dom_ids[0], np);

                                    // find domain in my list of patterns
                                    int k=0;
                                    for (const auto& pat : my_patterns)
                                    {
                                        if (pat.domain_id() == dom_ids[j]) break;
                                        ++k;
                                    }
                                    auto& pat = my_patterns[k];

                                    // recv all iteration spaces for each domain and each pair
                                    for (const auto& did : send_dom_ids)
                                    {
                                        int num_is;
                                        comm.recv(rank, 0, num_is);
                                        std::vector<iteration_space> is(num_is);
                                        comm.recv(rank, 0, &is[0], num_is);
                                        auto& vec = pat.send_halos()[did];
                                        vec.insert(vec.end(), is.begin(), is.end());
                                    }
                                    ++j;
                                }

                            }
                        }
                    }
                }

                for (int r=0; r<(int)num_domain_ids.size(); ++r)
                {
                    comm.barrier();
                    if (comm.rank()==r)
                    {
                        std::cout << "rank " << r << ": listing recv maps\n";
                        for (const auto& p : my_patterns)
                        {
                            std::cout << "  pattern for domain " << p.domain_id() << std::endl;
                            for (const auto& p0 : p.recv_halos())
                            {
                                    std::cout << "      extended_id: "
                                    << "id = " << p0.first.id << ", "
                                    << "rank = " << p0.first.mpi_rank << ", "
                                    << "address = " << p0.first.address << ", "
                                    << "tag = " << p0.first.tag << ", "
                                    << "num is = " << p0.second.size() << "\n"; 
                                for (const auto& is : p0.second)
                                {
                                    std::cout << "        {(" << is.first()[0] << ", " << is.first()[1] << ", " << is.first()[2] << "), "
                                                       << "(" << is.last()[0] << ", " << is.last()[1] << ", " << is.last()[2] << ")}"
                                                       << std::endl;
                                }
                            }
                        }
                    }
                    comm.barrier();
                    if (comm.rank()==r)
                    {
                        std::cout << "rank " << r << ": listing send maps\n";
                        for (const auto& p : my_patterns)
                        {
                            std::cout << "  pattern for domain " << p.domain_id() << std::endl;
                            for (const auto& p0 : p.send_halos())
                            {
                                    std::cout << "      extended_id: "
                                    << "id = " << p0.first.id << ", "
                                    << "rank = " << p0.first.mpi_rank << ", "
                                    << "address = " << p0.first.address << ", "
                                    << "tag = " << p0.first.tag << ", "
                                    << "num is = " << p0.second.size() << "\n"; 
                                for (const auto& is : p0.second)
                                {
                                    std::cout << "        {(" << is.first()[0] << ", " << is.first()[1] << ", " << is.first()[2] << "), "
                                                       << "(" << is.last()[0] << ", " << is.last()[1] << ", " << is.last()[2] << ")}"
                                                       << std::endl;
                                }
                            }
                        }
                    }
                    comm.barrier();
                }

                return pattern_container<P,grid_type,domain_id_type>(std::move(my_patterns));
            }
        };

    } // namespace detail

} // namespace gridtools

#endif /* INCLUDED_STRUCTURED_PATTERN_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

