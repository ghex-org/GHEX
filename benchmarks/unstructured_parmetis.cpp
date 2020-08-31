/* 
 * GridTools
 * 
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 * 
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 * 
 */


#include <cstdint>
#include <fstream>
#include <vector>

#include <mpi.h>

#include <gtest/gtest.h>

#include <parmetis.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/threads/std_thread/primitives.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/user_concepts.hpp>
// include communication object header file


#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif


template<typename T>
char* as_bytes(T& i) {
    return reinterpret_cast<char*>(&i); // TO DO: really, is it safe?
}


template<typename C>
void debug_print(const C& c) {
    std::cout << "Size = " << c.size() << "; elements = [ ";
    for (const auto& elem : c) { std::cout << elem << " "; }
    std::cout << "]\n";
}


/** @brief Unstructured exchange benchmark (in-place receive against buffered receive)*/
TEST(unstructured_parmetis, receive_type) {

    // type definitions
    using data_int_type = int64_t;

    // MPI setup
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // input data
    std::ifstream ap_fs("Ap.out", std::ios_base::binary);
    ap_fs.seekg(0, std::ios_base::end); // go to the end

    // TO DO: debug info, please remove
    if (rank == 0) {
        std::cout << "number of int64_t read: " << ap_fs.tellg() / sizeof(data_int_type) << "\n";
    } // output is 817 (CSR format provides always the two endpoints, first included, second excluded)

    data_int_type all_num_vertices = ap_fs.tellg() / sizeof(data_int_type) - 1;
    auto pos = all_num_vertices / size * sizeof(data_int_type) * rank;
    ap_fs.seekg(pos); // rewind to begin of section, according to rank (remainder is handled entirely by last rank, TO DO: not optimal)
    if (rank == (size - 1)) { // last rank reads until eof
        std::vector<data_int_type> ap{};
        for (data_int_type b; ap_fs.read(as_bytes(b), sizeof(b)); ) {
            ap.push_back(b);
        }
    } else { // all other ranks read until end of their section // HERE
        data_int_type section_size = all_num_vertices / size + 1; // (CSR format provides always the two endpoints, first included, second excluded)
        std::vector<data_int_type> ap(section_size);
    }

    // std::ifstream ai_fs("Ai.out", std::ios_base::binary);

    // std::vector<data_int_type> ai{};
    // for (data_int_type i; ai_fs.read(as_bytes(i), sizeof(i));) { ai.push_back(i); } // TO DO: set bounds according to rank

    /* // TO DO: debug info, please remove
    std::cout << "Ap:\n";
    debug_print(ap);
    std::cout << "\n";
    std::cout << "Ai:\n";
    debug_print(ai); */

    // prepare parmetis routine arrays

    // call parmetis routine

    // repartition output according to parmetis labelling

    // GHEX context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    // GHEX user concepts

}
