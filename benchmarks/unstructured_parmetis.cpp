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

#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
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
#include <ghex/arch_list.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/pattern.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/unstructured/communication_object_ipr.hpp>
#include <ghex/common/timer.hpp>
#include <ghex/common/accumulator.hpp>


// GHEX type definitions
#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#else
using transport = gridtools::ghex::tl::ucx_tag;
using threading = gridtools::ghex::threads::std_thread::primitives;
#endif
using domain_id_type = int;
using global_index_type = idx_t;
using domain_descriptor_type = gridtools::ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = gridtools::ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = gridtools::ghex::unstructured::grid;
template<typename T>
using data_descriptor_cpu_type = gridtools::ghex::unstructured::data_descriptor<gridtools::ghex::cpu, domain_id_type, global_index_type, T>;
using timer_type = gridtools::ghex::timer;


template<typename T>
char* as_bytes(T& i) {
    return reinterpret_cast<char*>(&i); // TO DO: really, is it safe?
}

template<typename T, typename C>
std::vector<int> counts_as_bytes(const C& c) {
    std::vector<int> res(c.size());
    std::transform(c.begin(), c.end(), res.begin(), [](auto i){ return i * sizeof(T); });
    return res;
}

std::vector<int> counts_to_displs(const std::vector<int>& counts) {
    std::vector<int> displs(counts.size(), 0);
    for (std::size_t i = 1; i < counts.size(); ++i) {
        displs[i] = displs[i-1] + counts[i-1];
    }
    return displs;
}

template<typename Domain, typename Field, typename O>
void initialize_field(const Domain& d, Field& f, O d_id_offset) {
    using value_type = typename Field::value_type;
    assert(f.size() == d.size());
    for (std::size_t i = 0; i < d.inner_size(); ++i) {
        f[i] = static_cast<value_type>(d.domain_id()) * d_id_offset + static_cast<value_type>(d.vertices()[i]);
    }
}

template<typename Domain, typename Pattern, typename Field, typename O>
void check_exchanged_data(const Domain& d, const Pattern& p, const Field& f, O d_id_offset) {
    using domain_id_type = typename Domain::domain_id_type;
    using index_type = typename Pattern::index_type;
    using value_type = typename Field::value_type;
    std::map<index_type, domain_id_type> halo_map{}; // index -> recv_domain_id
    for (const auto& rh : p.recv_halos()) {
        for (const auto i : rh.second.front().local_indices()) {
            halo_map.insert(std::make_pair(i, rh.first.id));
        }
    }
    for (const auto& pair : halo_map) {
        value_type actual = f[pair.first];
        value_type expected = static_cast<value_type>(pair.second) * d_id_offset + static_cast<value_type>(d.vertices()[pair.first]);
        EXPECT_EQ(actual, expected);
    }
}

template<typename Domain, typename Pattern>
Domain make_reindexed_domain(const Domain& d, const Pattern& p) {
    using vertices_type = typename Domain::vertices_type;
    vertices_type vs{};
    vs.reserve(d.size());
    vs.insert(vs.end(), d.vertices().begin(), d.vertices().begin() + d.inner_size());
    for (const auto& rh : p.recv_halos()) {
        for (const auto i : rh.second.front().local_indices()) {
            vs.push_back(d.vertices()[i]);
        }
    }
    Domain res{d.domain_id(), vs, d.inner_size(), d.levels()}; // TO DO: std::move vs?
    return res;
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

    // ========== repartition output according to parmetis labeling ==========

    // 1) vertices distribution map
    using vertices_dist_type = std::map<int, std::map<idx_t, std::vector<idx_t>>>;
    vertices_dist_type vertices_dist{};
    for (idx_t v_id = vtxdist_v[rank], i = 0; i < static_cast<idx_t>(ap_n.size() - 1); ++v_id, ++i) {
        vertices_dist[part_v[i]].insert(std::make_pair(v_id, std::vector<idx_t>{ai.begin() + ap_n[i], ai.begin() + ap_n[i+1]}));
    }

    // 2) all-to-all: number of vertices per rank
    std::vector<int> s_n_vertices_rank(size);
    for (int i = 0; i < size; ++i) {
        s_n_vertices_rank[i] = vertices_dist[i].size(); // any missing rank gets actually inserted into the map here
    };
    std::vector<int> r_n_vertices_rank(size);
    MPI_Alltoall(s_n_vertices_rank.data(), sizeof(int), MPI_BYTE,
                 r_n_vertices_rank.data(), sizeof(int), MPI_BYTE,
                 comm);

    // 3) all-to-all: vertex ids
    std::vector<idx_t> s_v_ids_rank{};
    s_v_ids_rank.reserve(ap_n.size() - 1);
    for (const auto& r_m_pair : vertices_dist) {
        for (const auto& v_a_pair : r_m_pair.second) {
            s_v_ids_rank.push_back(v_a_pair.first);
        }
    }
    std::vector<int> s_v_ids_rank_counts = counts_as_bytes<idx_t>(s_n_vertices_rank);
    std::vector<int> s_v_ids_rank_displs = counts_to_displs(s_v_ids_rank_counts);
    std::vector<idx_t> r_v_ids_rank(std::accumulate(r_n_vertices_rank.begin(), r_n_vertices_rank.end(), 0));
    std::vector<int> r_v_ids_rank_counts = counts_as_bytes<idx_t>(r_n_vertices_rank);
    std::vector<int> r_v_ids_rank_displs = counts_to_displs(r_v_ids_rank_counts);
    MPI_Alltoallv(s_v_ids_rank.data(), s_v_ids_rank_counts.data(), s_v_ids_rank_displs.data(), MPI_BYTE,
                  r_v_ids_rank.data(), r_v_ids_rank_counts.data(), r_v_ids_rank_displs.data(), MPI_BYTE,
                  comm);

    // 4) all-to-all: adjacency size per vertex per rank
    std::vector<int> s_adjncy_size_vertex_rank{};
    s_adjncy_size_vertex_rank.reserve(ap_n.size() - 1);
    for (const auto& r_m_pair : vertices_dist) {
        for (const auto& v_a_pair : r_m_pair.second) {
            s_adjncy_size_vertex_rank.push_back(v_a_pair.second.size());
        }
    }
    std::vector<int> s_adjncy_size_vertex_rank_counts = counts_as_bytes<int>(s_n_vertices_rank);
    std::vector<int> s_adjncy_size_vertex_rank_displs = counts_to_displs(s_adjncy_size_vertex_rank_counts);
    std::vector<int> r_adjncy_size_vertex_rank(std::accumulate(r_n_vertices_rank.begin(), r_n_vertices_rank.end(), 0));
    std::vector<int> r_adjncy_size_vertex_rank_counts = counts_as_bytes<int>(r_n_vertices_rank);
    std::vector<int> r_adjncy_size_vertex_rank_displs = counts_to_displs(r_adjncy_size_vertex_rank_counts);
    MPI_Alltoallv(s_adjncy_size_vertex_rank.data(), s_adjncy_size_vertex_rank_counts.data(), s_adjncy_size_vertex_rank_displs.data(), MPI_BYTE,
                  r_adjncy_size_vertex_rank.data(), r_adjncy_size_vertex_rank_counts.data(), r_adjncy_size_vertex_rank_displs.data(), MPI_BYTE,
                  comm);

    // 5) all-to-all: adjacency per rank
    std::vector<idx_t> s_adjncy_rank{};
    s_adjncy_rank.reserve(std::accumulate(s_adjncy_size_vertex_rank.begin(), s_adjncy_size_vertex_rank.end(), 0));
    for (const auto& r_m_pair : vertices_dist) {
        for (const auto& v_a_pair : r_m_pair.second) {
            s_adjncy_rank.insert(s_adjncy_rank.end(), v_a_pair.second.begin(), v_a_pair.second.end());
        }
    }
    std::vector<int> s_adjncy_rank_counts{};
    s_adjncy_rank_counts.reserve(size);
    for (auto a_it = s_adjncy_size_vertex_rank.begin(), r_it = s_n_vertices_rank.begin(); r_it < s_n_vertices_rank.end(); ++r_it) {
        s_adjncy_rank_counts.push_back(std::accumulate(a_it, a_it + *r_it, 0) * sizeof(idx_t));
        a_it += *r_it;
    }
    std::vector<int> s_adjncy_rank_displs = counts_to_displs(s_adjncy_rank_counts);
    std::vector<idx_t> r_adjncy_rank(std::accumulate(r_adjncy_size_vertex_rank.begin(), r_adjncy_size_vertex_rank.end(), 0));
    std::vector<int> r_adjncy_rank_counts{};
    r_adjncy_rank_counts.reserve(size);
    for (auto a_it = r_adjncy_size_vertex_rank.begin(), r_it = r_n_vertices_rank.begin(); r_it < r_n_vertices_rank.end(); ++r_it) {
        r_adjncy_rank_counts.push_back(std::accumulate(a_it, a_it + *r_it, 0) * sizeof(idx_t));
        a_it += *r_it;
    }
    std::vector<int> r_adjncy_rank_displs = counts_to_displs(r_adjncy_rank_counts);
    MPI_Alltoallv(s_adjncy_rank.data(), s_adjncy_rank_counts.data(), s_adjncy_rank_displs.data(), MPI_BYTE,
                  r_adjncy_rank.data(), r_adjncy_rank_counts.data(), r_adjncy_rank_displs.data(), MPI_BYTE,
                  comm);

    // =======================================================================

    // GHEX context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int gh_rank = context.rank();
    auto gh_comm = context.get_communicator(context.get_token());

    // GHEX constants
    const idx_t d_id_offset = 10e9;
    const int n_iters = 50;

    // timers
    timer_type t_buf_local, t_buf_global; // 1 - unordered halos - buffered receive
    timer_type t_ord_buf_local, t_ord_buf_global; // 2 - ordered halos - buffered receive
    timer_type t_ord_ipr_local, t_ord_ipr_global; // 3 - ordered halos - in-place receive

    // output file
    std::stringstream ss_file;
    ss_file << gh_rank;
    std::string filename = "unstructured_parmetis_receive_type_" + ss_file.str() + ".txt";
    std::ofstream file(filename.c_str());
    file << "Unstructured ParMETIS receive type benchmark\n\n";

    // print sizes info
    idx_t n_vertices_local{r_v_ids_rank.size()}, n_vertices_global;
    idx_t n_edges_local{r_adjncy_rank.size()}, n_edges_global;
    MPI_Allreduce(&n_vertices_local, &n_vertices_global, 1, MPI_INT64_T, MPI_SUM, context.mpi_comm()); // MPI type set according to parmetis idx type
    MPI_Allreduce(&n_edges_local, &n_edges_global, 1, MPI_INT64_T, MPI_SUM, context.mpi_comm()); // MPI type set according to parmetis idx type
    file << "local vertices: " << n_vertices_local << "\n"
         << "global vertices: " << n_vertices_global << "\n"
         << "local edges: " << n_edges_local << "\n"
         << "global edges: " << n_edges_global << "\n\n";

    // 1 ======== unordered halos - buffered receive =========================

    // setup
    domain_id_type d_id{gh_rank}; // 1 domain per rank
    domain_descriptor_type d{d_id, r_v_ids_rank, r_adjncy_rank}; // CSR constructor
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};
    auto p = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);
    using pattern_container_type = decltype(p);
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(gh_comm);
    std::vector<idx_t> f(d.size(), 0);
    initialize_field(d, f, d_id_offset);
    data_descriptor_cpu_type<idx_t> data{d, f};

    // exchange
    auto h = co.exchange(p(data)); // first iteration
    h.wait();
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h = co.exchange(p(data));
        h.wait();
        t_local.toc();
        t_buf_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_buf_global(t_global);
    }

    // check
    check_exchanged_data(d, p[0], f, d_id_offset);

    // 2 ======== ordered halos - buffered receive ===========================

    // setup
    domain_descriptor_type d_ord = make_reindexed_domain(d, p[0]);
    std::vector<domain_descriptor_type> local_domains_ord{d_ord};
    auto p_ord = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains_ord); // TO DO: definitely not optimal, only recv halos are different
    auto co_ord = gridtools::ghex::make_communication_object<pattern_container_type>(gh_comm); // new one, same conditions
    std::vector<idx_t> f_ord(d_ord.size(), 0);
    initialize_field(d_ord, f_ord, d_id_offset);
    data_descriptor_cpu_type<idx_t> data_ord{d_ord, f_ord};

    // exchange
    auto h_ord = co_ord.exchange(p_ord(data_ord)); // first iteration
    h_ord.wait();
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_ord = co_ord.exchange(p_ord(data_ord));
        h_ord.wait();
        t_local.toc();
        t_ord_buf_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ord_buf_global(t_global);
    }

    // check
    check_exchanged_data(d_ord, p_ord[0], f_ord, d_id_offset);

    // 3 ======== ordered halos - in-place receive ===========================

    // setup
    auto co_ipr = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(gh_comm);
    std::vector<idx_t> f_ipr(d_ord.size(), 0);
    initialize_field(d_ord, f_ipr, d_id_offset);
    data_descriptor_cpu_type<idx_t> data_ipr{d_ord, f_ipr};

    // exchange
    auto h_ipr = co_ipr.exchange(p_ord(data_ipr)); // first iteration
    h_ipr.wait();
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_ipr = co_ipr.exchange(p_ord(data_ipr));
        h_ipr.wait();
        t_local.toc();
        t_ord_ipr_local(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ord_ipr_global(t_global);
    }

    // check
    check_exchanged_data(d_ord, p_ord[0], f_ipr, d_id_offset);

    // ======== output =======================================================

    file << "1 - unordered halos - buffered receive\n"
         << "\tlocal time = " << t_buf_local.mean() / 1000.0 << "+/-" << t_buf_local.stddev() / 1000.0 << "s\n"
         << "\tglobal time = " << t_buf_global.mean() / 1000.0 << "+/-" << t_buf_global.stddev() / 1000.0 << "s\n";

    file << "2 - ordered halos - buffered receive\n"
         << "\tlocal time = " << t_ord_buf_local.mean() / 1000.0 << "+/-" << t_ord_buf_local.stddev() / 1000.0 << "s\n"
         << "\tglobal time = " << t_ord_buf_global.mean() / 1000.0 << "+/-" << t_ord_buf_global.stddev() / 1000.0 << "s\n";

    file << "3 - ordered halos - in-place receive\n"
         << "\tlocal time = " << t_ord_ipr_local.mean() / 1000.0 << "+/-" << t_ord_ipr_local.stddev() / 1000.0 << "s\n"
         << "\tglobal time = " << t_ord_ipr_global.mean() / 1000.0 << "+/-" << t_ord_ipr_global.stddev() / 1000.0 << "s\n";

    // MPI setup
    MPI_Comm_free(&comm);

}
