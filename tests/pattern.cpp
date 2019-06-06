// 
// GridTools
// 
// Copyright (c) 2014-2019, ETH Zurich
// All rights reserved.
// 
// Please, refer to the LICENSE file in the root directory.
// SPDX-License-Identifier: BSD-3-Clause
// 
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
            for (int i=is.first()[0]; i<=is.last()[0]; ++i)
            for (int j=is.first()[1]; j<=is.last()[1]; ++j)
            for (int k=is.first()[2]; k<=is.last()[2]; ++k)
            {
                std::cout << "packing [" << i << ", " << j << ", " << k << "] at " << (void*)(&(buffer[counter])) << std::endl;
                buffer[counter++] = T(100);
            }
        }
    }

    template<typename IndexContainer>
    void unpack(const T* buffer, const IndexContainer& c)
    {
        for (const auto& is : c)
        {
            T res;
            std::size_t counter=0;
            for (int i=is.first()[0]; i<=is.last()[0]; ++i)
            for (int j=is.first()[1]; j<=is.last()[1]; ++j)
            for (int k=is.first()[2]; k<=is.last()[2]; ++k)
            {
                std::cout << "unpacking [" << i << ", " << j << ", " << k << "] : " << buffer[counter] << std::endl;
                res = buffer[counter++];
            }
        }
    }

    int m_dom_id;
};

bool test0(boost::mpi::communicator& mpi_comm)
{
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};

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
            std::vector<typename my_domain_desc::box> halos;
            typename my_domain_desc::box bottom{ d.first(), d.last() };

            bottom.m_last[2]   = bottom.m_first[2]-1;
            bottom.m_first[2] -= 2;
            bottom.m_first[2]  = (bottom.m_first[2]+20)%20;
            bottom.m_last[2]   = (bottom.m_last[2]+20)%20;

            halos.push_back( bottom );

            auto top{bottom};
            top.m_first[2] = 0;
            top.m_last[2]  = 1;

            halos.push_back( top );

            typename my_domain_desc::box left{ d.first(), d.last() };
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

            halos.push_back( right );

            return halos;
        };

    auto halo_gen2 = [&mpi_comm](const my_domain_desc& d)
        {
            std::vector<typename my_domain_desc::box> halos;
            typename my_domain_desc::box bottom{ d.first(), d.last() };

            bottom.m_last[2]   = bottom.m_first[2]-1;
            bottom.m_first[2] -= 2;
            bottom.m_first[2]  = (bottom.m_first[2]+20)%20;
            bottom.m_last[2]   = (bottom.m_last[2]+20)%20;

            halos.push_back( bottom );

            auto top{bottom};
            top.m_first[2] = 0;
            top.m_last[2]  = 1;

            halos.push_back( top );
            
            typename my_domain_desc::box left{ d.first(), d.last() };
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

            halos.push_back( right );

            return halos;
        };

    //auto patterns = gridtools::make_pattern<gridtools::protocol::mpi, gridtools::structured_grid>(comm, halo_gen, local_domains);
    
    auto patterns1 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen1, local_domains);
    auto patterns2 = gridtools::make_pattern<gridtools::structured_grid>(mpi_comm, halo_gen2, local_domains);

    using pattern_t = typename decltype(patterns1)::value_type;
    gridtools::communication_object_erased<gridtools::protocol::mpi,typename pattern_t::grid_type,int> co_erased;

    //gridtools::communication_object<typename decltype(patterns1)::value_type> co;

    my_field<double, gridtools::device::cpu> field1_a{0};
    my_field<double, gridtools::device::gpu> field1_b{1};

    my_field<int, gridtools::device::cpu> field2_a{0};
    my_field<int, gridtools::device::gpu> field2_b{1};

    /*co.exchange(
        patterns1(field1_a),
        patterns1(field1_b),
        patterns2(field2_a),
        patterns2(field2_b)
    );*/

    auto h = co_erased.exchange(
        patterns1(field1_a),
        patterns1(field1_b),
        patterns2(field2_a),
        patterns2(field2_b)
    );

    h.post();

    h.wait();

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

