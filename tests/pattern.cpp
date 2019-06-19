// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
#include "../include/simple_field.hpp"
#include "../include/structured_pattern.hpp"
#include "../include/communication_object_erased.hpp"
#include <boost/mpi/environment.hpp>
#include <array>

struct my_domain_desc
{
    using coordinate_type = std::array<int,3>;
    using domain_id_type  = int;

    struct box
    {
        const coordinate_type& first() const { return m_first; }
        const coordinate_type& last() const { return m_last; }
        coordinate_type m_first;
        coordinate_type m_last;
    };

    struct box2
    {
        const box& local() const { return m_local; }
        const box& global() const { return m_global; }
        box m_local;
        box m_global;
    };

    domain_id_type domain_id() const { return m_id; }
    const coordinate_type& first() const { return m_first; }
    const coordinate_type& last() const { return m_last; }

    domain_id_type  m_id;
    coordinate_type m_first;
    coordinate_type m_last;
};

template<typename T, typename Device>
struct my_field
{
    using value_type = T;
    
    using device_type = Device;

    typename device_type::id_type device_id() const { return 0; }

    int domain_id() const { return m_dom_id; }

    template<typename IndexContainer>
    void pack(T* buffer, const IndexContainer& c)
    {
        for (const auto& is : c)
        {
            std::size_t counter=0;
            for (int i=is.local().first()[0]; i<=is.local().last()[0]; ++i)
            for (int j=is.local().first()[1]; j<=is.local().last()[1]; ++j)
            for (int k=is.local().first()[2]; k<=is.local().last()[2]; ++k)
            {
                //std::cout << "packing [" << i << ", " << j << ", " << k << "] at " << (void*)(&(buffer[counter])) << std::endl;
                buffer[counter++] = T(i);
            }
        }
    }

    template<typename IndexContainer>
    void unpack(const T* buffer, const IndexContainer& c)
    {
        for (const auto& is : c)
        {
            //T res;
            std::size_t counter=0;
            for (int i=is.local().first()[0]; i<=is.local().last()[0]; ++i)
            for (int j=is.local().first()[1]; j<=is.local().last()[1]; ++j)
            for (int k=is.local().first()[2]; k<=is.local().last()[2]; ++k)
            {
                std::cout << "unpacking [" << i << ", " << j << ", " << k << "] : " << buffer[counter] << std::endl;
                //T res = buffer[counter++];
                ++counter;
            }
        }
    }

    int m_dom_id;
};

bool test0(boost::mpi::communicator& mpi_comm)
{
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};

    const std::array<int,3> g_first{0,0,0};
    const std::array<int,3> g_last{39, ((comm.size()-1)/2+1)*15-1, 19};
    const std::array<int,3> local_ext{10,15,20};
    const std::array<int,3> offset{3,3,3};
    const int max_memory = (local_ext[0]+2*offset[0])*(local_ext[1]+2*offset[1])*(local_ext[2]+2*offset[2]);

    std::vector<double> field_1a_raw(max_memory);
    std::vector<double> field_1b_raw(max_memory);
    std::vector<int> field_2a_raw(max_memory);
    std::vector<int> field_2b_raw(max_memory);

    using domain_descriptor_type = gridtools::simple_domain_descriptor<int,3>;
    std::vector<domain_descriptor_type> local_domains_;
    local_domains_.push_back( domain_descriptor_type{
        comm.rank()*2, 
        std::array<int,3>{ (comm.rank()%2)*20,       (comm.rank()/2)*15,  0},
        std::array<int,3>{ (comm.rank()%2)*20+9, (comm.rank()/2+1)*15-1, 19}});
    local_domains_.push_back( domain_descriptor_type{
        comm.rank()*2+1,
        std::array<int,3>{ (comm.rank()%2)*20+10,     (comm.rank()/2)*15,  0},
        std::array<int,3>{ (comm.rank()%2)*20+19, (comm.rank()/2+1)*15-1, 19}});

    auto halo_gen1_ = domain_descriptor_type::halo_generator_type(
        g_first, g_last,
        {1,1,1,1,1,1}, 
        {true,true,true});
    auto halo_gen2_ = domain_descriptor_type::halo_generator_type(
        g_first, g_last,
        {2,2,2,2,2,2}, 
        {true,true,true});
    auto pattern1_ = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen1_, local_domains_);
    auto pattern2_ = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen2_, local_domains_);

    gridtools::simple_field_wrapper<double,gridtools::device::cpu,domain_descriptor_type, 2,1,0> field_1a_{
        local_domains_[0].domain_id(),
        field_1a_raw.data(),
        offset,local_ext};
    gridtools::simple_field_wrapper<double,gridtools::device::cpu,domain_descriptor_type, 2,1,0> field_1b_{
        local_domains_[1].domain_id(),
        field_1b_raw.data(),
        offset,local_ext};
    gridtools::simple_field_wrapper<int,gridtools::device::cpu,domain_descriptor_type, 2,1,0> field_2a_{
        local_domains_[0].domain_id(),
        field_2a_raw.data(),
        offset,local_ext};
    gridtools::simple_field_wrapper<int,gridtools::device::cpu,domain_descriptor_type, 2,1,0> field_2b_{
        local_domains_[1].domain_id(),
        field_2b_raw.data(),
        offset,local_ext};

    auto co_       = gridtools::make_communication_object(pattern1_,pattern2_);

    co_.bexchange(
        pattern1_(field_1a_),
        pattern1_(field_1b_),
        pattern2_(field_2a_),
        pattern2_(field_2b_));



    std::vector<my_domain_desc> local_domains;

    local_domains.push_back(
        my_domain_desc{ 
            comm.rank()*2,
            typename my_domain_desc::coordinate_type{ (comm.rank()%2)*20,     (comm.rank()/2)*15,  0},
            typename my_domain_desc::coordinate_type{ (comm.rank()%2)*20+9, (comm.rank()/2+1)*15-1, 19} } );

    local_domains.push_back(
        my_domain_desc{ 
            comm.rank()*2+1,
            typename my_domain_desc::coordinate_type{ (comm.rank()%2)*20+10,     (comm.rank()/2)*15,  0},
            typename my_domain_desc::coordinate_type{ (comm.rank()%2)*20+19, (comm.rank()/2+1)*15-1, 19} } );

    auto halo_gen1 = [&mpi_comm](const my_domain_desc& d)
        {
            std::vector<typename my_domain_desc::box2> halos;
            /*typename my_domain_desc::box bottom{ d.first(), d.last() };

            bottom.m_last[2]   = bottom.m_first[2]-1;
            bottom.m_first[2] -= 2;
            bottom.m_first[2]  = (bottom.m_first[2]+20)%20;
            bottom.m_last[2]   = (bottom.m_last[2]+20)%20;

            halos.push_back( bottom );

            auto top{bottom};
            top.m_first[2] = 0;
            top.m_last[2]  = 1;

            halos.push_back( top );*/

            typename my_domain_desc::box left{ d.first(), d.last() };
            left.m_last[0]   = left.m_first[0]-1; 
            left.m_first[0] -= 2;
            left.m_first[0]  = (left.m_first[0]+40)%40;
            left.m_last[0]   = (left.m_last[0]+40)%40;
            typename my_domain_desc::box left_l{left};
            left_l.m_first[0] = -2;
            left_l.m_last[0]  = -1;
            left_l.m_first[1] =  0;
            left_l.m_last[1]  =  d.last()[1]-d.first()[1];
            left_l.m_first[2] =  0;
            left_l.m_last[2]  =  d.last()[2]-d.first()[2];

            halos.push_back( typename my_domain_desc::box2{left_l,left} );


            typename my_domain_desc::box right{ d.first(), d.last() };
            right.m_first[0]  = right.m_last[0]+1; 
            right.m_last[0]  += 2;
            right.m_first[0]  = (right.m_first[0]+40)%40;
            right.m_last[0]   = (right.m_last[0]+40)%40;
            typename my_domain_desc::box right_l{left};
            right_l.m_first[0] =  d.last()[0]-d.first()[0]+1;
            right_l.m_last[0]  =  d.last()[0]-d.first()[0]+2;
            right_l.m_first[1] =  0;
            right_l.m_last[1]  =  d.last()[1]-d.first()[1];
            right_l.m_first[2] =  0;
            right_l.m_last[2]  =  d.last()[2]-d.first()[2];

            halos.push_back( typename my_domain_desc::box2{right_l,right} );


            return halos;
        };

    //auto halo_gen2 = halo_gen1;
    auto halo_gen2 = [&mpi_comm](const my_domain_desc& d)
        {
            std::vector<typename my_domain_desc::box2> halos;
            typename my_domain_desc::box bottom{ d.first(), d.last() };
            bottom.m_last[2]   = bottom.m_first[2]-1;
            bottom.m_first[2] -= 2;
            bottom.m_first[2]  = (bottom.m_first[2]+20)%20;
            bottom.m_last[2]   = (bottom.m_last[2]+20)%20;
            typename my_domain_desc::box bottom_l{bottom};
            bottom_l.m_first[2] = -2;
            bottom_l.m_last[2]  = -1;
            bottom_l.m_first[0] =  0;
            bottom_l.m_last[0]  =  d.last()[0]-d.first()[0];
            bottom_l.m_first[1] =  0;
            bottom_l.m_last[1]  =  d.last()[1]-d.first()[1];

            halos.push_back( typename my_domain_desc::box2{bottom_l,bottom} );

            auto top{bottom};
            top.m_first[2] = 0;
            top.m_last[2]  = 1;
            typename my_domain_desc::box top_l{top};
            top_l.m_first[2] =  d.last()[2]-d.first()[2]+1;
            top_l.m_last[2]  =  d.last()[2]-d.first()[2]+2;
            top_l.m_first[0] =  0;
            top_l.m_last[0]  =  d.last()[0]-d.first()[0];
            top_l.m_first[1] =  0;
            top_l.m_last[1]  =  d.last()[1]-d.first()[1];

            halos.push_back( typename my_domain_desc::box2{top_l,top} );
            
            /*typename my_domain_desc::box left{ d.first(), d.last() };
            left.m_last[0]   = left.m_first[0]-1; 
            left.m_first[0] -= 2;
            left.m_first[0]  = (left.m_first[0]+40)%40;
            left.m_last[0]   = (left.m_last[0]+40)%40;

            halos.push_back( left );

            typename my_domain_desc::box right{ d.first(), d.last() };
            right.m_first[0]  = right.m_last[0]+1; 
            right.m_last[0]  += 2;
            right.m_first[0]  = (right.m_first[0]+40)%40;
            right.m_last[0]   = (right.m_last[0]+40)%40;

            halos.push_back( right );*/

            return halos;
        };

    //auto patterns = gridtools::make_pattern<gridtools::protocol::mpi, gridtools::structured_grid>(comm, halo_gen, local_domains);
    
    auto pattern1 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen1, local_domains);
    auto pattern2 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen2, local_domains);

    auto co       = gridtools::make_communication_object(pattern1,pattern2);

    my_field<double, gridtools::device::cpu> field1_a{0};
    my_field<double, gridtools::device::gpu> field1_b{1};

    my_field<int, gridtools::device::cpu> field2_a{0};
    my_field<int, gridtools::device::gpu> field2_b{1};

    co.bexchange(
        pattern1(field1_a),
        pattern1(field1_b),
        pattern2(field2_a),
        pattern2(field2_b));

    /*auto h = co.exchange(
        pattern1(field1_a),
        pattern1(field1_b),
        pattern2(field2_a),
        pattern2(field2_b)
    );

    h.wait();*/

    return true;
}

int main(int argc, char* argv[])
{
    //MPI_Init(&argc,&argv);
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;


    auto passed = test0(world);

    return 0;
}

// modelines
// vim: set ts=4 sw=4 sts=4 et: 
// vim: ff=unix: 

