#include "../include/pattern.hpp"
#include <boost/mpi/environment.hpp>
#include <iostream>
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


    auto halo_gen = [&mpi_comm](const my_domain_desc& d)
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

    auto patterns = gridtools::make_pattern<gridtools::protocol::mpi, gridtools::structured_grid>(comm, halo_gen, local_domains);

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
