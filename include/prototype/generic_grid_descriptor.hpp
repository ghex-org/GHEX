#ifndef INCLUDED_GENERIC_GRID_DESCRIPTOR_HPP
#define INCLUDED_GENERIC_GRID_DESCRIPTOR_HPP

#include <type_traits>



namespace gridtools {

// total dimensions
// dimension of decomposed domain
//    regular?
// remaining dimensions of domain
//    regular?
//

template<unsigned N, bool Regular = true>
struct dimensions
{
    using rank = std::integral_constant<unsigned, N>;
    using regular = std::integral_constant<bool, Regular>;
};


template<typename DecomposedDimensions, typename RemainingDimensions = dimensions<0>>
struct grid_topology
{
    using decomposed_dimensions = DecomposedDimensions;
    using remaining_dimensions  = RemainingDimensions;

    // total rank/dimensionality
    using rank = std::integral_constant<unsigned, decomposed_dimensions::rank::value + remaining_dimensions::rank::value>;

    // check if at least 3D
    static_assert(rank::value > 2, "not enough dimensions specified");


    // fully regular grid
    // ------------------
    using regular = std::integral_constant<bool, 
           decomposed_dimensions::regular::value 
        && remaining_dimensions::regular::value>;
    
    // check if decomposed in 2 dimensions at most
    static_assert(!regular::value || (decomposed_dimensions::rank::value > 0 && decomposed_dimensions::rank::value < 3),
        "decomposition can be at least 2D for this topology");

    // hypercube
    using hypercube = std::integral_constant<bool, regular::value && (rank::value > 3)>;


    // fully unstructured grid
    // -----------------------
    using unstructured = std::integral_constant<bool, 
           !decomposed_dimensions::regular::value 
        && remaining_dimensions::rank::value==0>;

    // check if decomposed in 3D
    static_assert(!unstructured::value || rank::value==3, "invalid unstructured grid");


    // mixed topology
    // --------------
    using mixed = std::integral_constant<bool, 
           !decomposed_dimensions::regular::value 
        && decomposed_dimensions::rank::value==2 
        && rank::value==3 
        && remaining_dimensions::regular::value>;

    // check if valid topology
    static_assert( regular::value || unstructured::value || mixed::value , "invalid topology specified");
    static_assert( 
           ( regular::value && !unstructured::value && !mixed::value)
        || (!regular::value &&  unstructured::value && !mixed::value)
        || (!regular::value && !unstructured::value &&  mixed::value)        
        , "invalid topology specified");
};


template<unsigned DD, unsigned RD = 1>
using regular_grid_topology = grid_topology< dimensions<DD>, dimensions<RD> >;

using mixed_grid_topology = grid_topology< dimensions<2,false>, dimensions<1> >;

using unstructured_grid_topology = grid_topology< dimensions<3,false>, dimensions<0> >;


template<unsigned Faces>
struct inflated_regular_topology : public regular_grid_topology<2,1>
{
    using nested = std::true_type;
    using face_topology = regular_grid_topology<2,1>;
    using faces = std::integral_constant<unsigned, Faces>;
};

using inflated_cube_topology = inflated_regular_topology<6>;

using inflated_icosahedron_topology = inflated_regular_topology<10>;

namespace _impl {

template<typename T, typename V = void>
struct _nested : std::false_type {};

template<typename T>
struct _nested<T, typename std::enable_if<T::nested::value, void>::type> : T::nested {};


template<typename T, typename V = void>
struct _faces : std::integral_constant<unsigned,0> {};

template<typename T>
struct _faces<T, typename std::enable_if<(T::faces::value > 0), void>::type> : T::faces {};
} // namespace _impl


template<typename Topology>
struct topology_traits
{
    using type = Topology;
    using rank   = typename type::rank;
    using decomposed_dimensions = typename type::decomposed_dimensions;
    using remaining_dimensions  = typename type::remaining_dimensions;
    using regular = typename type::regular;
    using mixed   = typename type::mixed;
    using unstructured = typename type::unstructured;
    using nested = typename _impl::_nested<type>;
    using faces  = typename _impl::_faces<type>;
};

} // namespace gridtools


#endif /* INCLUDED_GENERIC_GRID_DESCRIPTOR_HPP */
