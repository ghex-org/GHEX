


#include "../include/prototype/generic_grid_descriptor.hpp"
#include <iostream>

using namespace gridtools;


// regular grid
using reg_dim_4 = dimensions<4,true>;
using reg_dim_3 = dimensions<3,true>;
using reg_dim_2 = dimensions<2,true>;
using reg_dim_1 = dimensions<1,true>;
using reg_dim_0 = dimensions<0,true>;
using regular_topology_12 = grid_topology<reg_dim_1,reg_dim_2>;
using regular_topology_21 = grid_topology<reg_dim_2,reg_dim_1>;
using regular_topology_13 = grid_topology<reg_dim_1,reg_dim_3>;
using regular_topology_22 = grid_topology<reg_dim_2,reg_dim_2>;
using regular_topology_14 = grid_topology<reg_dim_1,reg_dim_4>;
using regular_topology_23 = grid_topology<reg_dim_2,reg_dim_3>;

// unstructured grid
using dim_4 = dimensions<4,false>;
using dim_3 = dimensions<3,false>;
using dim_2 = dimensions<2,false>;
using dim_1 = dimensions<1,false>;
using unstructured_topology = grid_topology<dim_3>;
using unstructured_topology_21 = grid_topology<dim_2,dim_1>;
using unstructured_topology_4 = grid_topology<dim_4>;

// mixed topology
using mixed_topology = grid_topology<dim_2,reg_dim_1>;



int main()
{
    regular_topology_12 r_12;
    regular_topology_21 r_21;
    regular_topology_13 r_13;
    regular_topology_22 r_22;
    regular_topology_14 r_14;
    regular_topology_23 r_23;

    unstructured_topology u;

    //mixed_topology m;

    regular_grid_topology<1,2> rgt0;
    regular_grid_topology<2,1> rgt1;

    unstructured_grid_topology u_;

    mixed_grid_topology m_;


    using inflated_topology = nested_topology<2,1>;

    std::cout << inflated_topology::rank::value << std::endl;

    std::cout << topology_traits<regular_grid_topology<1,2>>::nested::value << std::endl;
    std::cout << topology_traits<inflated_topology>::nested::value << std::endl;

    return 0;
}
