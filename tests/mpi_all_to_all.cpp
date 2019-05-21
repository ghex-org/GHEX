
#include "../include/protocol/mpi.hpp"
#include <boost/mpi/environment.hpp>
#include <iostream>

template<typename T>
bool test_0(boost::mpi::communicator& mpi_comm)
{
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};
    std::vector<T> values;
    {
        auto f = comm.all_gather( static_cast<T>(comm.address()) );
        values = f.get();
    }
    for (const auto& v : values)
        std::cout << v << " ";
    std::cout << std::endl;
    return true;
}

template<typename T>
bool test_1(boost::mpi::communicator& mpi_comm)
{
    gridtools::protocol::communicator<gridtools::protocol::mpi> comm{mpi_comm};
    int my_num_values = (comm.address()+1)*2;
    std::vector<T> my_values(my_num_values);
    for (int i=0; i<my_num_values; ++i)
        my_values[i] = (comm.address()+1)*1000 + i;
    auto num_values = comm.all_gather(my_num_values).get();
    auto values = comm.all_gather(my_values, num_values).get();
    for (const auto& vec : values)
        for (const auto& v : vec)
            std::cout << v << " ";
    std::cout << std::endl;
    return true; 
}

int main(int argc, char* argv[])
{
    //MPI_Init(&argc,&argv);
    boost::mpi::environment env(argc, argv);
    boost::mpi::communicator world;

    auto passed = test_0<int>(world);
    passed = passed && test_0<double>(world);

    passed = passed && test_1<int>(world);
    passed = passed && test_1<double>(world);

    return 0;
}
