#include <mpi.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <list>
#include <type_traits>
#include <utility>
#include <array>
#include <vector>
#include <thread>
#include <algorithm>
#include <prototype/generic_interfaces.hpp>

using id_type = int; // this could be unsigned

struct dir_type {
    int value;

    dir_type(int v) : value{v} {}

    operator int() const { return value; }
    static int direction2int(dir_type d) {return d.value+1;}
    static int invert_direction(dir_type d) {return -d.value;}
};

template <int N, typename T>
void show_data(T x, std::ostream& s = std::cout) {
    s << "Data:\n";
    for (int i = 0; i<N; ++i) {
            s << x[i] << ", ";
    }
    s << "\n";
    s.flush();
}


int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);


    int world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    std::stringstream ss;
    ss << rank;

    std::string filename = "out" + ss.str() + ".txt";

    std::cout << filename << std::endl;
    std::ofstream file(filename.c_str());

    /*
      We need to define the domain decomposition. The sub-domains are
      associated to processing grid (pg) nodes, so we need to define
      the topology of the processing grid
    */

    // local ids: We need a container of local ids (right now) even though this example has 1 id per MPI rank
    std::list<id_type> local_ids{ rank };

    { // DIAGNOSTIC BEGIN
        file << "Local ids\n";
        std::for_each(local_ids.begin(), local_ids.end(), [&file] (id_type const& x) { file << x << ", ";});
        file << "\n";
        file.flush();
    }

    // Given an id of a sub-domain we need the container of the neighbors of this id
    auto neighbor_generator = [world_size](id_type id) -> std::array<std::pair<id_type, dir_type>, 2> // this lambda returns a sequence of neighbors of a local_id
        {
            return { std::make_pair(mod(id-1, world_size), -1), {mod(id+1, world_size), 1} };
        };

    { // DIAGNOSTIC BEGIN
        file << "Local ids\n";
        std::for_each(local_ids.begin(), local_ids.end(), [&file, neighbor_generator] (id_type const& x) {
                auto list = neighbor_generator(x);
                file << "neighbors of ID = " << x << ":\n";
                std::for_each(list.begin(), list.end(), [&file](std::pair<id_type, dir_type> const& y) {file << y.first << ", "; });
                file << "\n";
            });
        file << "\n";
        file.flush();
    }

    // Generating the PG with the sequence of local IDs and a function to gather neighbors
    generic_pg<id_type, dir_type> pg( local_ids,
                            neighbor_generator,
                            file
                            );


    { // DIAGNOSTIC BEGIN
        pg.show_topology(file);
        file.flush();
        file.close();
    }

    // Iteration spaces describe there the data to send and data to
    // receive. An iteration space for a communication object is a
    // function that takes the local id and the remote id (should it
    // take just the remote id, since the local is just the one to
    // which it is associated with?), and return the region to
    // pack/unpack. For regula grids this will take the iteration
    // ranges prototyped in some file here.
    auto iteration_spaces_send = [](id_type local, id_type remote, dir_type direction) {struct A { int begin() const {return 1;}; int end(){return 2;}}; return A{};}; // send buffer for id
    auto iteration_spaces_recv = [](id_type local, id_type remote, dir_type direction) {
        struct A {
            dir_type d;
            A(dir_type d) : d{d} {}
            int begin() const {
                return d==1?2:0;
            }
            int end(){
                return d==1?3:1;
            }

        };
        return A{direction};
    }; // recv buffer from id

    // constructing the communication object with the id associated to it and the topology information (processinf grid)
    using co_type = generic_co<generic_pg<id_type, dir_type>, decltype(iteration_spaces_send), decltype(iteration_spaces_recv) >;
    // We need a CO object for each sub-domain in each rank
    std::vector<co_type> co;


    for (auto id : local_ids) {
        co.push_back(co_type{id, pg, iteration_spaces_send, iteration_spaces_recv});
    }

    // launching the computations [not finished yet, I adventured into a complex case and it does not work yet
    std::vector<std::thread> threads;
    auto itc = co.begin();
    auto id = *(local_ids.begin());
    {
        auto& co = *itc;
        id_type data[3] = {-1,id+1,-1};

        std::stringstream ss;
        ss << rank;
        ss << "-";
        ss << std::this_thread::get_id();

        std::string filename = "tout" + ss.str() + ".txt";

        std::cout << filename << std::endl;
        std::ofstream tfile(filename.c_str());
        tfile << "\nFILE for " << id << "\n";
        show_data<3>(data, tfile);
        tfile << "================================\n";

        auto hdl = co.exchange(data, tfile);
        hdl.wait();

        show_data<3>(data, tfile);
        tfile << mod(id-1, world_size)+1 << ", "
              << id+1 << ", "
              << mod(id+1, world_size)+1 << " <---- Should be\n";
        tfile << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n";

        if (mod(id-1, world_size)+1 == data[0] and id+1 == data[1] and mod(id+1, world_size)+1 == data[2]) {
            std::cout << id << ": result PASSED\n";
        } else {
            std::cout << id << " result FAIL\n";
        }
        safer_output("thread");
    }

    MPI_Finalize();
}
