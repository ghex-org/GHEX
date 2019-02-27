/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

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
#include <prototype/regular_grid_descriptors.hpp>

namespace gt = gridtools;

using id_type = std::pair<int, int>;
//using dir_type = std::pair<int, int>;

struct dir_type : public gt::direction<2> {

    using gt::direction<2>::direction;

    static int direction2int(dir_type d) {
        return d.m_data[0]*3+d.m_data[1];
    }

    static dir_type invert_direction(dir_type d) {
        return dir_type{std::array<int, 2>{-d.m_data[0], -d.m_data[1]}};
    }
};


namespace std {
    template<> struct hash<id_type> {
        std::size_t operator()(id_type const& t) const {
            return std::hash<int>{}(t.first);
        }
    };
}


template <int N, typename T>
void show_data(T x, std::ostream& s = std::cout) {
    s << "Data:\n";
    for (int i = 0; i<N; ++i) {
        for (int j = 0; j<N; ++j) {
            s << x[i][j] << ", ";
        }
        s << "\n";
    }
    s << "\n";
    s.flush();
}


class data_descriptor_t {
public:

    struct range {
        int m_begin;
        int m_end;

        range(int b, int e)
            : m_begin(b)
            , m_end(e)
        {}

        int begin() const {
            return m_begin;
        }

        int end() const {
            return m_end;
        }
    };

    static constexpr int rank = 3;

    std::array<int, rank> m_sizes;

    template <typename ...Sizes>
    data_descriptor_t(Sizes... s) : m_sizes{s...} {}

    template <int I>
    int begin() const {
        return 0;
    }

    template <int I>
    int end() const {
        return m_sizes[I];
    }

    template <int I>
    range range_of() const {
        return range(begin<I>(), end<I>());
    }
};


int main(int argc, char** argv) {
    int p;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &p);
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int grid_sizes[2] = {0,0};

    MPI_Dims_create(world_size, 2, grid_sizes);

    show(grid_sizes[0]);
    show(grid_sizes[1]);


    std::stringstream ss;
    ss << rank;

    std::string filename = "out" + ss.str() + ".txt";

    std::cout << filename << std::endl;
    std::ofstream file(filename.c_str());

    /*
      In this example we take a 2D regular domain and split it so that
      each MPI rank has two subdomains. This is a depiction of the
      decomposition in the case of 6 MPI ranks organized in a 3x2 grid.

             P0       ||         P1
      ------  ------  ||   ------  ------
      |0,0 |  |0,1 |  ||   |0,2 |  |0,3 |
      ------  ------  ||   ------  ------
   ___________________||___________________
             P2       ||         P3
      ------  ------  ||   ------  ------
      |1,0 |  |1,1 |  ||   |1,2 |  |1,3 |
      ------  ------  ||   ------  ------
   ___________________||___________________
             P4       ||          P5
      ------  ------  ||   ------  ------
      |2,0 |  |2,1 |  ||   |2,2 |  |2,3 |
      ------  ------  ||   ------  ------
     */

    // local ids: there are two domains per node, and they are identified by a pair of indices <i,j>
    std::list<id_type> local_ids{ {rank/(grid_sizes[1]), rank%(grid_sizes[1])*2}, {rank/(grid_sizes[1]), rank%(grid_sizes[1])*2+1} };

    file << "Local ids\n";
    std::for_each(local_ids.begin(), local_ids.end(), [&file] (id_type const& x) { file << x << ", ";});
    file << "\n";
    auto neighbor_generator = [grid_sizes](id_type id) -> std::array<std::pair<id_type, dir_type>, 4> // this lambda returns a sequence of neighbors of a local_id
        {
            int i = id.first;
            int j = id.second;
            return { std::make_pair(id_type{mod(i-1, grid_sizes[0]),j}, std::array<int, 2>{-1,0}), {{mod(i+1, grid_sizes[0]),j}, std::array<int, 2>{1,0}},  {{i,mod(j-1,grid_sizes[1]*2)}, std::array<int, 2>{0,-1}}, {{i,mod(j+1,grid_sizes[1]*2)}, std::array<int, 2>{0,1}} };
        };

    file << "Local ids\n";
    std::for_each(local_ids.begin(), local_ids.end(), [&file, neighbor_generator] (id_type const& x) {
            auto list = neighbor_generator(x);
            file << "neighbors of ID = " << x << ":\n";
            std::for_each(list.begin(), list.end(), [&file](std::pair<id_type, dir_type> const& y) {file << y.first << ", "; });
            file << "\n";
        });
    file << "\n";

    // Generating the PG with the sequence of local IDs and a function to gather neighbors
    generic_pg<id_type, dir_type> pg( local_ids,
                                      neighbor_generator,
                                      file
                                      );


    pg.show_topology(file);
    file.flush();
    file.close();

    data_descriptor_t data_dsc(3);
    std::array<gt::halo_sizes, 2> halos = { gt::halo_sizes{ 1, 1 }, gt::halo_sizes{ 1, 1 } };

    gt::regular_grid_descriptor< 2 /* number of partitioned dimensions */ > grid(halos);


    // Iteration spaces describe there the data to send and data to
    // receive. An iteration space for a communication object is a
    // function that takes the local id and the remote id (should it
    // take just the remote id, since the local is just the one to
    // which it is associated with?), and return the region to
    // pack/unpack. For regula grids this will take the iteration
    // ranges prototyped in some file here.
    auto iteration_spaces_send = [&data_dsc, &grid](id_type local, id_type remote, dir_type direction) {
        return grid.inner_iteration_space< gt::partitioned<0, 1> >(data_dsc, direction);

    };

    auto iteration_spaces_recv = [&data_dsc, &grid](id_type local, id_type remote, dir_type direction) {
        return grid.outer_iteration_space< gt::partitioned<0, 1> >(data_dsc, direction);
    };

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
    for (auto it = local_ids.begin(); it != local_ids.end(); ++it, ++itc) {
        threads.push_back(std::thread([&file, pg, rank, grid_sizes](id_type id, co_type& co) -> void
                                      {
                                          id_type data[3][3] = { {{-1,-1}, {-1,-1}, {-1,-1}},
                                                                 {{-1,-1},    id  , {-1,-1}},
                                                                 {{-1,-1}, {-1,-1}, {-1,-1}}};

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
                                          tfile << "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^\n\n";
                                          tfile.flush();
                                          safer_output("thread");
                                          tfile.close();

                                          auto r01 = std::pair<int, int>{mod(id.first-1, grid_sizes[0]), id.second};
                                          auto r10 = std::pair<int, int>{id.first, mod(id.second-1, grid_sizes[1]*2)};
                                          auto r21 = std::pair<int, int>{mod(id.first+1, grid_sizes[0]), id.second};
                                          auto r12 = std::pair<int, int>{id.first, mod(id.second+1, grid_sizes[1]*2)};


                                          if (data[0][1] == r01 and data[1][0] == r10 and data[2][1] == r21 and data[1][2] == r12) {
                                              std::cout << id << ": PASSED\n";
                                          } else {
                                              std::cout << id << ": FAILED\n";
                                          }
                                      },
                                      *it, std::ref(*itc)));
    }

    for (auto& th : threads) {
        th.join();
    }

    MPI_Finalize();
}
