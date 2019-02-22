#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <unordered_map>
#include <list>
#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <thread>
#include <atomic>
#include <algorithm>
#include <mutex>
#include <cassert>

const int log_max_objs = 10; // log_2(max number of PGs)

inline int mod(int __i, int __j) { return (((((__i % __j) < 0) ? (__j + __i % __j) : (__i % __j)))); }

std::atomic<int> object_count{1}; // set to one to see the effect with a singleobject - 0 is perfectly fine

int rank;

std::mutex io_serialize;

template <typename T, typename U>
std::ostream& operator<<(std::ostream& s, std::pair<T,U> x) {
    return s << "(" << x.first << ", " << x.second << ")";
}

void safer_output(std::string const& s, std::ostream& stream = std::cout) {
    std::lock_guard<std::mutex> lock(io_serialize);
    stream << rank << ": " << s << "\n";
    stream.flush();
}

#define show(x) safer_output( #x + std::string(" ") + std::to_string(x))


template <typename UniqueId, typename RankId>
struct node_uid : std::pair<UniqueId, RankId> {
public:
    using base_t = std::pair<UniqueId, RankId>;
    using std::pair<UniqueId, RankId>::pair;

    UniqueId unique_id() const { return base_t::first; }
    UniqueId& unique_id() { return base_t::first; }
    RankId rank() const { return base_t::second; }
    RankId& rank() { return base_t::second; }
};

template <typename UniqueId, typename RankId>
std::ostream& operator<<(std::ostream& s, node_uid<UniqueId, RankId> i) {
    return s << "< uid: " << i.first << ", rank: " << i.second << " >";
}

template <typename DomainId, typename DirectionId, typename RemoteId>
struct neighbor_spec;

template <typename DomainId, typename DirectionId, typename UniqueId, typename RankId>
class neighbor_spec<DomainId, DirectionId, node_uid<UniqueId, RankId>>
{
    using data_type = std::tuple<DomainId, DirectionId, node_uid<UniqueId, RankId > >;
    data_type data;
 public:
    neighbor_spec(DomainId did, DirectionId dirid, node_uid<UniqueId, RankId > nuid)
        : data{did, dirid, nuid}
    {}

    DomainId id() const {return std::get<0>(data);}
    DomainId& id() {return std::get<0>(data);}

    DirectionId direction() const {return std::get<1>(data);}
    DirectionId& direction() {return std::get<1>(data);}

    node_uid<UniqueId, RankId> uid() const {return std::get<2>(data);}
    node_uid<UniqueId, RankId>& uid() {return std::get<2>(data);}
};

template <typename Container>
struct neighbor_list: public Container {
    using base_t = Container;
    using base_t::base_t;
    using neighbor_spec_t = typename Container::value_type;
};

template <typename NeighborList, typename UniqueID>
class node_info : std::pair<NeighborList, UniqueID > {
public:
    using base_t = std::pair<NeighborList, UniqueID >;
    using base_t::base_t;

    using list_type = NeighborList;

    NeighborList const& list() const {return base_t::first;}
    NeighborList& list() {return base_t::first;}
    UniqueID uid() const {return base_t::second;}
    UniqueID& uid() {return base_t::second;}
};

/**
   A computing grid, dubbed pg, is an object that contains topological
   information about neighbors of a domain, where a domain is a piece of
   the domain decomposed application specific chunk of data
 */
template <typename DomainId, typename DirectionId>
class generic_pg {

public:
    using direction_type = DirectionId;

    using unique_id_t = int;
    using rank_id_t = int;

    using node_uid_t = node_uid<unique_id_t, rank_id_t>;
    using neighbor_list_t = neighbor_list<std::list<neighbor_spec<DomainId, DirectionId, node_uid_t> > >;
    using node_info_t = node_info<neighbor_list_t, node_uid_t >;
private:
    static node_uid_t invalid_node_uid() { return {-1,-1}; }

    // Map from a domain ID to a pair made of a list of neighbors+theirID and the unique integer identifier of this ID across the system
    using topology_t = std::unordered_map< DomainId, node_info_t >;

    /**
       Given the local ids and the neighbor generator, we can store
       the local information about the topology in a hash-map
     */
    template <typename LD, typename G>
    topology_t fill_topology(LD const& ld, G const& g) {
        static_assert(std::is_convertible<typename LD::value_type, DomainId>::value, "");
        topology_t topology;
        for (auto id : ld) {
            auto neighbors = g(id);
            typename topology_t::mapped_type::list_type n_list;
            using neighbor_spec_type = typename topology_t::mapped_type::list_type::value_type;
            for (std::pair<DomainId, DirectionId> neighbor: neighbors) {
                n_list.push_back(neighbor_spec_type(neighbor.first, neighbor.second, invalid_node_uid())); // These will be filled out later - need communication
            }
            topology[id] = node_info_t{std::move(n_list), invalid_node_uid()}; // These will be filled out later - need communication
        }

        return std::move(topology);
    }

    topology_t m_topology;
    int m_rank = -1;
    int obj_index = -1; // unique id of the PG object. Since they are
                        // created in other ranks with in the same
                        // order, this counter should be consistent
                        // with all other ranks.

public:
    using id_type = DomainId;

    // We need to generate some unique identifiers for the messages we
    // will send out. This is done using object_count, which is an
    // atomic counter.
    template <typename LocalDomains, typename Generator>
    generic_pg(LocalDomains const& ld, Generator const& g, std::ostream& outs)
        : m_topology(fill_topology(ld, g)), obj_index(object_count++)
    {
        MPI_Comm_rank(MPI_COMM_WORLD, &m_rank);

        {   // Check that the obj_index is the same in all ranks. The
            // order of construction of the objects should be the same
            // everywhere otherwise we are in trouble
            int world_size;
            MPI_Comm_size(MPI_COMM_WORLD, &world_size);
            std::vector<int> obj_ids(world_size);
            MPI_Allgather(&obj_index, 1, MPI_INT, (void*)obj_ids.data(), 1, MPI_INT, MPI_COMM_WORLD);
            bool ok = true;
            std::for_each(obj_ids.begin(), obj_ids.end(), [&ok, this](int& x) {ok = ok and (obj_index == x); });
            assert(ok);
        }

        int send_size = m_topology.size();
        int total_size;
        MPI_Allreduce(&send_size, &total_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

        show(total_size);

        int offset;
        MPI_Scan(&send_size, &offset, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        offset -= m_topology.size();

        show(offset);

        // The scan above is telling where to start putting the information in

        std::for_each(m_topology.begin(), m_topology.end(),
                      [&offset, this](typename topology_t::value_type & pair) {
                          pair.second.uid().unique_id() = (obj_index << log_max_objs) + offset++; pair.second.uid().rank() = m_rank;});


        std::vector<std::pair<DomainId, node_uid_t > > temp;
        std::for_each(m_topology.begin(), m_topology.end(),
                      [&temp](typename topology_t::value_type & pair)
                      {
                          temp.push_back(std::make_pair(pair.first /*The local id - key */, node_uid_t{pair.second.uid().unique_id(), pair.second.uid().rank() }));
                      }
                      );

        std::vector< int > all_sizes(total_size);
        MPI_Allgather(&send_size, 1, MPI_INT, (void*)all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);
        std::for_each(all_sizes.begin(), all_sizes.end(), [](int& v) { v *= sizeof(std::pair<DomainId, node_uid_t >); });
        std::vector< int > all_offsets(total_size);
        for (int i = 1; i < total_size; ++i) {
            all_offsets[i] = all_offsets[i-1] + all_sizes[i-1];
        }

        std::vector<std::pair<DomainId, node_uid_t > > all_ids(total_size);
        MPI_Allgatherv((int*)(void*)temp.data(), temp.size()*sizeof(std::pair<DomainId, node_uid_t >), MPI_CHAR, (void*)all_ids.data(), all_sizes.data(), all_offsets.data(), MPI_CHAR, MPI_COMM_WORLD);

        for (int i = 0; i < total_size; ++i) {
            auto const& x = (reinterpret_cast< std::pair<DomainId, node_uid_t >* >(all_ids.data())[i]);
            for (auto & map_elem : m_topology) {
                for (auto & lst_elem : map_elem.second.list()) {
                    if (lst_elem.id() == x.first) {
                        lst_elem.uid().unique_id() = x.second.unique_id();
                        lst_elem.uid().rank() = x.second.rank();
                    }
                }
            }
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    struct show_node_pair {
        std::ostream& s;

        show_node_pair(std::ostream& s) : s{s} {}

        //template <typename T>
        void operator()(typename topology_t::value_type const & pair) const
        {
            s << "|Node ID: " << pair.first << "\n"
              << "|\t -> global ID: " + std::to_string(pair.second.uid().unique_id()) << ", "
              << "Rank: " + std::to_string(pair.second.uid().rank()) << "\n"
              << "|\t Neighbors with IDs: ";

            std::for_each(pair.second.list().begin(), pair.second.list().end(),
                          [this](typename topology_t::value_type::second_type::list_type::value_type const& p)
                          {
                              s << "{" << p.id() << ", " << p.uid() << "}, ";
                          });
            s << "\n";
            s << "\n";
        }
    };

    struct show_node_info {
        std::ostream& s;

        show_node_info(std::ostream& s) : s{s} {}

        //template<typename T>
        void operator()(node_info_t const/*typename topology_t::value_type*/ & ninfo) const
        {
            s << "\t -> global ID: " + std::to_string(ninfo.uid().unique_id()) << ", "
              << "Rank: " + std::to_string(ninfo.uid().rank()) << "\n"
              << "\t Neighbors with IDs: ";

            std::for_each(ninfo.list().begin(), ninfo.list().end(),
                          [this](typename node_info_t::list_type::value_type const& p)
                          {
                              s << "{" << p.id() << ", " << p.uid() << "}, ";
                          });
            s << "\n";
            s << "\n";
        }
    };

    template < typename Stream = std::ostream >
    void show_topology(Stream &s = std::cout) {
        s << "Topology printout begin\n";
        std::for_each(m_topology.begin(), m_topology.end(), show_node_pair(s));
        s << "\n";
        s << "Topology printout end\n";
    }


    int unique_id(DomainId id) const {
        return m_topology.at(id).second.first;
    }

    int rank_of_id(DomainId id) const {
        return m_topology.at(id).second.second;
    }

    node_info_t const& id_info(DomainId const id) const {
        return m_topology.at(id);
    }


    node_uid_t const& neighbor_info(DomainId const id, DomainId const neighbor_id) const {
        auto l = m_topology[id].fisrt;
        auto it = std::find_if(l.begin(), l.end(),
                               [neighbor_id](std::pair<DomainId, node_uid_t > & elem)
                               {
                                   return elem.first == neighbor_id;
                               });
        assert(it != l.end());
        return *it;
    }

    int neighbor_rank(DomainId const id, DomainId const neighbor_id) const {
        return neigbor_info(id, neighbor_id).second.second;
    }

    int neighbor_unique_id(DomainId const id, DomainId const neighbor_id) const {
        return neigbor_info(id, neighbor_id).second.first;
    }
};


/**
   A Communication Object (CO) contains the information of what data
   had to be send to and received from, where the sources and
   destinations are provided by the Processing Grid (PG), shown above.

   IterationSpaces are templates here butthey should not. HAving them
   here allows to store the functions without knowing what their
   return type are. To be fixed in future, but this is a prototype.
 */
template <typename PG, typename IterationSpacesSend, typename IterationSpacesRecv>
class generic_co {
    using id_type = typename PG::id_type;
    using direction_type = typename PG::direction_type;

    id_type m_id;
    PG const& m_pg;
    IterationSpacesSend m_send_iteration_space;
    IterationSpacesRecv m_recv_iteration_space;

    struct future {
        std::vector<MPI_Request> request;

        future(std::vector<MPI_Request> x) : request{std::move(x)} {}

        void wait() {
            MPI_Status st;
            for (auto& r : request) {
                if (r != 0) MPI_Wait(&r, &st);
            }
        }
    };


    int tag(int proc_id, int unique_id) {
        return (proc_id << 20) + unique_id;
    }

public:
    generic_co(id_type id, PG const& pg,  IterationSpacesSend send_iteration_space,  IterationSpacesRecv recv_iteration_space)
        : m_id{id}
        , m_pg{pg}
        , m_send_iteration_space(send_iteration_space)
        , m_recv_iteration_space(recv_iteration_space)
    {}

    generic_co(generic_co const&) = delete;
    generic_co(generic_co&&) = default;

    id_type domain_id() const { return m_id; };

    template <typename D>
    future exchange(D* data, std::ostream& fl) {
        using TT = typename std::remove_all_extents<typename std::remove_pointer<D>::type>::type;

        auto const& info = m_pg.id_info(m_id);

        typename PG::show_node_info{fl}(info);

        auto my_unique_id = info.uid().unique_id();
        auto my_rank = info.uid().rank();

        std::vector<MPI_Request> request(info.list().size());

        std::for_each(request.begin(), request.end(), [](MPI_Request const &x) { std::cout << "R>" << x << "< "; });
        std::cout << "\n";

        int ind=0;
        std::for_each(info.list().begin(), info.list().end(),
                      [&fl, my_unique_id, my_rank, &ind, &request, data, this] (typename std::remove_reference<decltype(info.list())>::type::value_type const& neighbor)
                      {
                          /** this is a very sketchy firt exaxmple -
                              buffers are contiguous and tags are
                              wrong (tags should not identigy
                              neighbors, but messages
                              (neighbot+directon). */
                          auto r = m_recv_iteration_space(m_id, neighbor.id(), neighbor.direction());

                          //if (my_rank == neighbor.uid().rank()) return;

                          fl << m_id << ": Recv " << sizeof(TT)*(r.end()-r.begin()) << " bytes from "
                             << neighbor.uid().rank() << " (id: " << neighbor.id() << ") "
                             << " with tag " << my_unique_id<<7 + direction_type::direction2int(direction_type::invert_direction(neighbor.direction()))
                             << " recv in: " << r.begin()
                             << "\n";
                          fl.flush();
                          MPI_Irecv((reinterpret_cast<TT*>(reinterpret_cast<char*>(data)+r.begin()*sizeof(TT))),
                                    sizeof(TT)*(r.end()-r.begin()), MPI_CHAR, neighbor.uid().rank(),
                                    my_unique_id<<7 + direction_type::direction2int(direction_type::invert_direction(neighbor.direction())), MPI_COMM_WORLD, &request[ind++]);
                          fl << "Done " << ind << "\n";
                          fl.flush();
                      }
                      );

        std::for_each(info.list().begin(), info.list().end(),
                      [&fl, my_unique_id, my_rank, &request, data, this] (typename std::remove_reference<decltype(info.list())>::type::value_type const& neighbor)
                      {
                          /** this is a very sketchy firt exaxmple -
                              buffers are contiguous and tags are
                              wrong (tags should not identigy
                              neighbors, but messages
                              (neighbot+directon). */
                          auto s = m_send_iteration_space(m_id, neighbor.id(), neighbor.direction());

                          MPI_Status st;
                          MPI_Request mock;

                          //if (my_rank == neighbor.uid().rank()) return;

                          fl << m_id << ": Send " << sizeof(TT)*(s.end()-s.begin()) << " bytes to   "
                             << neighbor.uid().rank() << " (id: " << neighbor.id() << ") "
                             << " with tag " << neighbor.uid().unique_id()<<7 + direction_type::direction2int(neighbor.direction())
                             << " send from: " << s.begin()
                             << "\n";
                          fl.flush();
                          MPI_Isend((reinterpret_cast<TT*>(reinterpret_cast<char*>(data)+s.begin()*sizeof(TT))),
                                    sizeof(TT)*(s.end()-s.begin()), MPI_CHAR, neighbor.uid().rank(),
                                    neighbor.uid().unique_id()<<7 + direction_type::direction2int(neighbor.direction()), MPI_COMM_WORLD, &mock);
                          fl << "Done\n";
                          fl.flush();
                          MPI_Wait(&mock, &st);
                     }
                      );

        return request;

    }
};
