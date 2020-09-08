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
#include <array>
#include <algorithm>

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
    static_assert(std::is_same<data_int_type, idx_t>::value, "data integer type must be the same as ParMETIS integer type");

    // MPI setup
    MPI_Comm comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &comm);
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Ap
    std::ifstream ap_fs("Ap.out", std::ios_base::binary);
    ap_fs.seekg(0, std::ios_base::end); // go to the end
    idx_t all_num_vertices = ap_fs.tellg() / sizeof(idx_t) - 1;
    ap_fs.seekg(all_num_vertices / size * sizeof(idx_t) * rank); // rewind to begin of section, according to rank (remainder is handled entirely by last rank, TO DO: not optimal)
    std::vector<idx_t> ap{};
    if (rank == (size - 1)) { // last rank reads until eof
        for (idx_t b; ap_fs.read(as_bytes(b), sizeof(b)); ) {
            ap.push_back(b);
        }
    } else { // all other ranks read until end of their section
        idx_t section_size = all_num_vertices / size + 1; // (CSR format provides always the two endpoints, first included, second excluded)
        for (idx_t i = 0, b; i < section_size; ++i) {
            ap_fs.read(as_bytes(b), sizeof(b));
            ap.push_back(b);
        }
    }
    idx_t ap_offset = ap.front();
    std::vector<idx_t> ap_n(ap.size());
    std::transform(ap.begin(), ap.end(), ap_n.begin(), [ap_offset](auto i){ return i - ap_offset; }); // normalize

    // Ai
    std::ifstream ai_fs("Ai.out", std::ios_base::binary);
    ai_fs.seekg(ap.front() * sizeof(idx_t));
    std::vector<idx_t> ai{};
    for (idx_t i = ap.front(), b; i < ap.back(); ++i) {
        ai_fs.read(as_bytes(b), sizeof(b));
        ai.push_back(b);
    }

    // Vertices initial distribution
    std::vector<idx_t> vtxdist_v(size + 1);
    idx_t num_vertices = all_num_vertices / size;
    for (int i = 0; i < size; ++i) {
        vtxdist_v[i] = num_vertices * i;
    }
    vtxdist_v[size] = all_num_vertices;

    // Vertices final distribution (output)
    std::vector<idx_t> part_v(ap.size() - 1);

    // ParMETIS variables
    idx_t wgtflag = 0;
    idx_t numflag = 0;
    idx_t ncon = 1; // TO DO: double check
    idx_t nparts = size;
    std::vector<real_t> tpwgts_v(ncon * nparts, 1 / static_cast<real_t>(nparts)); // TO DO: double check
    std::vector<real_t> ubvec_v(ncon, 1.05); // TO DO: double check
    std::array<idx_t, 3> options{0, 0, 0};
    idx_t edgecut;

    // ParMETIS graph partitioning
    ParMETIS_V3_PartKway(vtxdist_v.data(),
                         ap_n.data(),
                         ai.data(),
                         NULL,
                         NULL,
                         &wgtflag,
                         &numflag,
                         &ncon,
                         &nparts,
                         tpwgts_v.data(),
                         ubvec_v.data(),
                         options.data(),
                         &edgecut,
                         part_v.data(),
                         &comm);

    // repartition output according to parmetis labeling

    using vertices_dist_type = std::map<int, std::map<idx_t, std::vector<idx_t>>>;
    vertices_dist_type vertices_dist{};
    for (idx_t v_id = vtxdist_v[rank], i = 0; i < ap_n.size() - 1; ++v_id, ++i) {
        vertices_dist[part_v[i]].insert(std::make_pair(v_id, std::vector<idx_t>{ai.begin() + ap_n[i], ai.begin() + ap_n[i+1]}));
    }



    // GHEX context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;

    // GHEX user concepts

    MPI_Comm_free(&comm);

}
