#ifndef INCLUDED_PATTERN_HPP
#define INCLUDED_PATTERN_HPP

#include "coordinate.hpp"
#include "protocol/communicator_base.hpp"
#include "protocol/mpi.hpp"
#include <map>

namespace gridtools {

template<typename P, typename GridType, typename DomainIdType>
class pattern
{};


template<typename CoordinateArrayType> 
struct _structured_grid 
{
    using coordinate_base_type    = CoordinateArrayType;
    using coordinate_type         = coordinate<coordinate_base_type>;
    using coordinate_element_type = typename coordinate_type::element_type;
    using dimension               = typename coordinate_type::dimension;    
};

struct structured_grid 
{
    template<typename Domain>
    using type = _structured_grid<typename Domain::coordinate_type>;
};

template<typename P, typename CoordinateArrayType, typename DomainIdType>
class pattern<P,_structured_grid<CoordinateArrayType>,DomainIdType>
{
public: // member types

    using grid_type               = _structured_grid<CoordinateArrayType>;
    using coordinate_type         = typename grid_type::coordinate_type;
    using coordinate_element_type = typename grid_type::coordinate_element_type;
    using dimension               = typename grid_type::dimension;
    using communicator_type       = protocol::communicator<P>;
    using address_type            = typename communicator_type::address_type;
    using communicator_mpi_type   = protocol::communicator<protocol::mpi>;
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

        bool operator<(const extended_domain_id_type& other) const noexcept { return id < other.id; }
    };

    using map_type = std::map<extended_domain_id_type, std::vector<iteration_space>>;

private: // members

    map_type m_send_map;
    map_type m_recv_map;

public: // ctors

public: // member functions

    const map_type& get_send_halos() const noexcept { return m_send_map; }
    const map_type& get_recv_halos() const noexcept { return m_recv_map; }
private: // implementation

};

namespace detail {

template<typename P, typename CoordinateArrayType, typename DomainRange>
auto make_pattern_impl(protocol::communicator<protocol::mpi>& comm, DomainRange&& d_range, _structured_grid<CoordinateArrayType>*)
{
    using domain_type       = typename std::remove_reference_t<DomainRange>::value_type;
    using domain_id_type    = typename domain_type::domain_id_type;
    using pattern_type      = pattern<P, _structured_grid<CoordinateArrayType>, domain_id_type>;

    return std::vector<pattern_type>{d_range.size()};

}

} // namespace detail

template<typename P, typename GridType, typename DomainRange>
auto make_pattern(protocol::communicator<protocol::mpi>& comm, DomainRange&& d_range)
{
    using grid_type = typename GridType::template type<typename std::remove_reference_t<DomainRange>::value_type>;
    return detail::make_pattern_impl<P>(comm, std::forward<DomainRange>(d_range), (grid_type*)0); 
}

} // namespace gridtools


#endif /* INCLUDED_PATTERN_HPP */

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

