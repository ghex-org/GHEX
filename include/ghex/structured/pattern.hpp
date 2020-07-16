/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */
#ifndef INCLUDED_GHEX_STRUCTURED_PATTERN_HPP
#define INCLUDED_GHEX_STRUCTURED_PATTERN_HPP

#include "./grid.hpp"
#include "../pattern.hpp"
#include <map>
#include <iosfwd>

namespace gridtools {
    namespace ghex {

    /** @brief structured pattern specialization
     *
     * This class provides access to the receive and send iteration spaces, determined by the halos, and holds
     * all connections to the neighbors.
     *
     * @tparam Communicator communicator type
     * @tparam CoordinateArrayType coordinate-like array type
     * @tparam DomainIdType domain id type*/
    template<typename Communicator, typename CoordinateArrayType, typename DomainIdType>
    class pattern<Communicator,structured::detail::grid<CoordinateArrayType>,DomainIdType>
    {
    public: // member types
        using grid_type               = structured::detail::grid<CoordinateArrayType>;
        using this_type               = pattern<Communicator, grid_type, DomainIdType>;
        using coordinate_type         = typename grid_type::coordinate_type;
        using coordinate_element_type = typename grid_type::coordinate_element_type;
        using dimension               = typename grid_type::dimension;
        using communicator_type       = Communicator;
        using address_type            = typename communicator_type::address_type;
        using domain_id_type          = DomainIdType;
        using pattern_container_type  = pattern_container<Communicator,grid_type,DomainIdType>;

        // this struct holds the first and the last coordinate (inclusive)
        // of a hypercube in N-dimensional space.
        // Invariant: first <= last
        struct iteration_space
        {
            using dimension = typename this_type::dimension;
        public: // member functions
                  coordinate_type& first()       noexcept { return _min; }
                  coordinate_type& last()        noexcept { return _max; }
            const coordinate_type& first() const noexcept { return _min; }
            const coordinate_type& last()  const noexcept { return _max; }

            // number of elements in hypercube
            int size() const noexcept 
            {
                int s = _max[0]-_min[0]+1;
                for (int i=1; i<coordinate_type::size(); ++i) s *= _max[i]-_min[i]+1;
                return s;
            }

        public: // members
            coordinate_type _min; 
            coordinate_type _max;

        public: // ctors
            iteration_space() noexcept = default;
            template<typename A, typename B>
            iteration_space(A&& a, B&& b)
            : _min(std::forward<A>(a)), _max(std::forward<B>(b))
            {
                if (!(a<=b)) throw std::runtime_error("iteration_space: first coordinate needs to be smaller or equal to the last coordinate");
            }
            iteration_space(const iteration_space&) = default;
            iteration_space(iteration_space&&) = default;

            iteration_space& operator=(const iteration_space& other) noexcept = default;
            iteration_space& operator=(iteration_space&& other) noexcept = default;

        public: // print
            template< class CharT, class Traits>
            friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const iteration_space& is)
            {
                os << "[" << is._min << ", " << is._max << "]";
                return os;
            }
        };

        // this struct combines local and global representation of a hypercube
        // and is a compostion of 2 iteration_space objects
        struct iteration_space_pair
        {
        public: // member types
            using pattern_type = pattern; 
            using dimension = typename this_type::dimension;

        public: // member functions
            iteration_space& local() noexcept { return m_local; }
            const iteration_space& local() const noexcept { return m_local; }
            iteration_space& global() noexcept { return m_global; }
            const iteration_space& global() const noexcept { return m_global; }
            int size() const noexcept { return m_local.size(); } 

        public: // members
            iteration_space m_local;
            iteration_space m_global;

        public: // print
            template< class CharT, class Traits>
            friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const iteration_space_pair& is)
            {
                os << is.m_global << " (local: " << is.m_local << ")";
                return os;
            }
        };

        // an extended domain id, including rank, address and tag information
        // used as key in halo lookup map
        struct extended_domain_id_type
        {
        public: // members
            domain_id_type id;
            int            mpi_rank;
            address_type   address;
            int            tag;

        public: // member functions
            // unique ordering given by id and tag
            bool operator<(const extended_domain_id_type& other) const noexcept 
            { 
                return (id < other.id ? true : (id == other.id ? (tag < other.tag) : false));
            }

        public: // print
            template< class CharT, class Traits>
            friend std::basic_ostream<CharT,Traits>& operator<<(std::basic_ostream<CharT,Traits>& os, const extended_domain_id_type& dom_id)
            {
                os << "{id=" << dom_id.id << ", tag=" << dom_id.tag << ", rank=" << dom_id.mpi_rank << "}";
                return os;
            }
        };

        // indices are stored as vector of iteration_space_pair objects
        using index_container_type = std::vector<iteration_space_pair>;
        // halo map type
        using map_type = std::map<extended_domain_id_type, index_container_type>;

    public: // static member function
        /** @brief compute number of elements in an object of type index_container_type */
        static int num_elements(const index_container_type& c) noexcept
        {
            int s = 0;
            for (const auto& is : c) s += is.size();
            return s;
        }

        friend class pattern_container<Communicator,grid_type,DomainIdType>;

    private: // members
        iteration_space_pair    m_domain;
        coordinate_type         m_global_first;
        coordinate_type         m_global_last;
        extended_domain_id_type m_id;
        map_type                m_send_map;
        map_type                m_recv_map;
        pattern_container_type* m_container;

    public: // ctors
        pattern(const iteration_space_pair& domain, const extended_domain_id_type& id)
        : m_domain(domain), m_id(id) {}
        pattern(const pattern&) = default;
        pattern(pattern&&) = default;

    public: // member functions
        map_type& send_halos() noexcept { return m_send_map; }
        const map_type& send_halos() const noexcept { return m_send_map; }
        map_type& recv_halos() noexcept { return m_recv_map; }
        const map_type& recv_halos() const noexcept { return m_recv_map; }
        domain_id_type domain_id() const noexcept { return m_id.id; }
        extended_domain_id_type extended_domain_id() const noexcept { return m_id; }
        const pattern_container_type& container() const noexcept { return *m_container; }
        coordinate_type& global_first() noexcept { return m_global_first; }
        coordinate_type& global_last()  noexcept { return m_global_last; }
        const coordinate_type& global_first() const noexcept { return m_global_first; }
        const coordinate_type& global_last()  const noexcept { return m_global_last; }
        const iteration_space& global_domain() const noexcept { return m_domain.global(); }

        /** @brief tie pattern to field
         * @tparam Field field type
         * @param field field instance
         * @return buffer_info object which holds pointers to the field and the pattern*/
        template<typename Field>
        buffer_info<pattern, typename Field::arch_type, Field> operator()(Field& field) const
        {
            return {*this,field,field.device_id()};
        }
    };

    namespace detail {

        // constructs the pattern with the help of all to all communications
        template<typename CoordinateArrayType>
        struct make_pattern_impl<::gridtools::ghex::structured::detail::grid<CoordinateArrayType>>
        {
            template<typename Transport, typename ThreadPrimitives, typename HaloGenerator, typename DomainRange>
            static auto apply(tl::context<Transport,ThreadPrimitives>& context, HaloGenerator&& hgen, DomainRange&& d_range)
            {
                // typedefs
                using context_type              = tl::context<Transport,ThreadPrimitives>;
                using domain_type               = typename std::remove_reference_t<DomainRange>::value_type;
                using domain_id_type            = typename domain_type::domain_id_type;
                using grid_type                 = ::gridtools::ghex::structured::detail::grid<CoordinateArrayType>;
                using communicator_type         = typename context_type::communicator_type;
                using pattern_type              = pattern<communicator_type, grid_type, domain_id_type>;
                using iteration_space           = typename pattern_type::iteration_space;
                using iteration_space_pair      = typename pattern_type::iteration_space_pair;
                using coordinate_type           = typename pattern_type::coordinate_type;
                using extended_domain_id_type   = typename pattern_type::extended_domain_id_type;

                // get this address from new communicator
                auto comm = context.get_setup_communicator();
                auto new_comm = context.get_serial_communicator();
                auto my_address = new_comm.address();
                
                // set up domain ids, extents and recv halos
                std::vector<iteration_space_pair>              my_domain_extents;
                std::vector<extended_domain_id_type>           my_domain_ids;
                std::vector<pattern_type>                      my_patterns;
                std::vector<std::vector<iteration_space_pair>> my_generated_recv_halos;
                // loop over domains and fill vectors with
                // - extended domain ids
                // - domain extents (local an global coordinates)
                // - pattern objects (one per domain)
                // - receive halos
                for (const auto& d : d_range)
                {
                    // fill data structures with domain related info
                    my_domain_ids.push_back( extended_domain_id_type{d.domain_id(), comm.rank(), my_address, 0} );
                    my_domain_extents.push_back( 
                        iteration_space_pair{
                            iteration_space{coordinate_type{d.first()}-coordinate_type{d.first()}, 
                                            coordinate_type{d.last()} -coordinate_type{d.first()}},
                            iteration_space{coordinate_type{d.first()}, coordinate_type{d.last()}}} );
                    my_patterns.emplace_back( /*new_comm,*/ my_domain_extents.back(), my_domain_ids.back() );
                    // make space for more halos
                    my_generated_recv_halos.resize(my_generated_recv_halos.size()+1);
                    // generate recv halos: invoke halo generator
                    auto recv_halos = hgen(d);
                    // convert the recv halos to internal format
                    for (const auto& h : recv_halos)
                    {
                        iteration_space_pair is{
                            iteration_space{coordinate_type{h.local().first()},coordinate_type{h.local().last()}},
                            iteration_space{coordinate_type{h.global().first()},coordinate_type{h.global().last()}}};
                        // check that invariant is fullfilled (halos are not empty)
                        if (is.local().first() <= is.local().last())
                        {
                            my_generated_recv_halos.back().push_back( is );
                        }
                    }
                }

                // find all domains and their extents by all_gather operations
                int my_num_domains   = my_domain_ids.size();
                auto num_domain_ids  = comm.all_gather(my_num_domains).get();
                auto domain_ids      = comm.all_gather(my_domain_ids, num_domain_ids).get();
                auto domain_extents  = comm.all_gather(my_domain_extents, num_domain_ids).get();
                const int world_size = num_domain_ids.size();

                // find global extents
                auto global_min = my_domain_extents[0].global().first();
                auto global_max = my_domain_extents[0].global().last();
                for (const auto& vec : domain_extents)
                    for (const auto& it_space_pair : vec)
                    {
                        global_min = min(global_min, it_space_pair.global().first());
                        global_max = max(global_max, it_space_pair.global().last());
                    }
                for (auto& pat : my_patterns)
                {
                    pat.global_first() = global_min;
                    pat.global_last()  = global_max;
                }
                
                // check my receive halos against all existing domains (i.e. intersection check)
                // in order to decide from which domain I shall be receiving from.
                // loop over patterns/domains
                auto d_it = std::begin(d_range);
                for (unsigned int i=0; i<my_patterns.size(); ++i)
                {
                    // get corresponding halos
                    const auto& recv_halos = my_generated_recv_halos[i];
                    // intersect each halo with all domain extents
                    for (const auto& halo : recv_halos)
                    {
                        // loop over all ranks
                        for (unsigned int j=0; j<domain_extents.size(); ++j)
                        {
                            // get vectors of extents and ids per rank
                            const auto& extents_vec   = domain_extents[j];
                            const auto& domain_id_vec = domain_ids[j];
                            // loop over extents and ids
                            for (unsigned int k=0; k<extents_vec.size(); ++k)
                            {
                                // intersect in global coordinates
                                const auto& extent = extents_vec[k];
                                const auto& domain_id = domain_id_vec[k];
                                const auto x = 
                                hgen.intersect(*d_it, halo.local().first(),    halo.local().last(),
                                                      halo.global().first(),   halo.global().last(),
                                                      extent.global().first(), extent.global().last());
                                const coordinate_type x_global_first{x.global().first()};
                                const coordinate_type x_global_last{x.global().last()};
                                if (x_global_first <= x_global_last) {
                                    my_patterns[i].recv_halos()[domain_id].push_back(
                                        iteration_space_pair{
                                            iteration_space{
                                                coordinate_type{x.local().first()},
                                                coordinate_type{x.local().last()}},
                                            iteration_space{x_global_first, x_global_last}});
                                }
                            }
                        }
                    }
                    ++d_it;
                }

                // set tags in order to disambiguate receives from same the PE
                // when dealing with multiple domains per PE
                int m_max_tag = 0;
                // map between MPI rank and maximum required tag
                std::map<int,int> tag_map;
                // loop over all patterns/domains
                for (auto& p : my_patterns)
                {
                    // loop over all halos
                    for (auto& id_is_pair : p.recv_halos())
                    {
                        // get rank from extended domain id
                        const int rank = id_is_pair.first.mpi_rank;
                        auto it = tag_map.find(rank);
                        if (it == tag_map.end())
                        {
                            // if neighbor rank is not yet in tag_map: insert it with maximum tag = 0
                            tag_map[rank] = 0;
                            // set tag in the receive halo data structure to 0
                            const_cast<extended_domain_id_type&>(id_is_pair.first).tag = 0;
                        }
                        else
                        {
                            // if neighbor rank is already present: increase the tag by 1
                            ++it->second;
                            m_max_tag = std::max(it->second,m_max_tag);
                            // set tag in the receive halo data structure
                            const_cast<extended_domain_id_type&>(id_is_pair.first).tag = it->second;
                        }
                    }
                }

                // communicate max tag to be used for thread safety in communication object 
                // use all_gather
                auto max_tags  = comm.all_gather(m_max_tag).get();
                // compute maximum tag and store in m_max_tag
                for (auto x : max_tags)
                    m_max_tag = std::max(x,m_max_tag);
                
                // translate my receive halos to (remote) send halos
                // by a detour over the following nested map
                std::map<int,
                    std::map<domain_id_type,
                        std::map<extended_domain_id_type, std::vector<iteration_space_pair> > > > send_halos_map;
                // which maps: rank -> remote domain id -> my domain's extended domain id -> vector of iteration spaces
                // loop over my patterns/domains
                for (const auto& p : my_patterns)
                {
                    // loop over all receive halos
                    for (const auto& id_is_pair : p.recv_halos())
                    {
                        // get my domain's default extended domain id (tag = 0)
                        auto d_id = p.extended_domain_id();
                        // change tag to the tag which was assigned before (tag = receive halo's tag) 
                        d_id.tag = id_is_pair.first.tag;
                        // create map entry if it does not exist
                        auto& is_vec = send_halos_map[id_is_pair.first.mpi_rank][id_is_pair.first.id][d_id];
                        // get number of previously added halo elements for this key
                        int s = is_vec.size();
                        // add new send halo to the list (copy the iteration spaces from receive halo)
                        is_vec.insert(is_vec.end(), id_is_pair.second.begin(), id_is_pair.second.end());
                        // recast local coordinates of the iteration spaces to remote's local coordinates
                        for (unsigned int l=s; l<is_vec.size(); ++l)
                        {
                            const auto& extents_vec = domain_extents[id_is_pair.first.mpi_rank];
                            const auto& domains_vec = domain_ids[id_is_pair.first.mpi_rank];
                            // find domain id
                            unsigned int ll=0; 
                            for (const auto& dd : domains_vec)
                            {
                                if (dd.id == id_is_pair.first.id) break;
                                ++ll;
                            }
                            // use iteration space's transform to do that?
                            const auto is_g = extents_vec[ll].global();
                            const auto is_l = extents_vec[ll].local();
                            is_vec[l].local().first() = is_l.first() + (is_vec[l].global().first()-is_g.first());
                            is_vec[l].local().last()  = is_l.first() + (is_vec[l].global().last()-is_g.first());
                        }
                    }
                }

                // communicate send halos to corresponding PEs

                // filter out my own send halos first
                auto it = send_halos_map.find(comm.rank());
                if (it != send_halos_map.end())
                {
                    // loop over temporary send halos for my rank
                    for (const auto& p1 : it->second)
                    {
                        const auto dom_id = p1.first;
                        unsigned int j = 0;
                        // find the right local domain
                        for (const auto& p : my_patterns)
                        {
                            if (p.domain_id() == dom_id) break;
                            ++j;
                        }
                        pattern_type& p = my_patterns[j];
                        // add send halos to the pattern
                        for (const auto& p2 : p1.second)
                            p.send_halos().insert(p2);
                    }
                    // remove the halos from temporary map
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
                                        std::vector<iteration_space_pair> is(num_is);
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

                return pattern_container<communicator_type,grid_type,domain_id_type>(std::move(my_patterns), m_max_tag);
            }

            template<typename Transport, typename ThreadPrimitives, typename HaloGenerator, typename RecvDomainIdsGen, typename DomainRange>
            static auto apply(tl::context<Transport,ThreadPrimitives>& context, HaloGenerator&& hgen, RecvDomainIdsGen&&, DomainRange&& d_range)
            {
                return apply(context, hgen, d_range);
            }
        };

    } // namespace detail
    } // namespace ghex

} // namespace gridtools

#endif /* INCLUDED_GHEX_STRUCTURED_PATTERN_HPP */

