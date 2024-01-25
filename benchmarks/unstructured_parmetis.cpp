/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <cstdlib>
#include <cassert>
#include <cstdint>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <algorithm>
#include <thread>
#include <cmath>

#include <mpi.h>

#include <gtest/gtest.h>

#include <parmetis.h>

#ifndef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/mpi/context.hpp>
#else
#include <ghex/transport_layer/ucx/context.hpp>
#endif
#include <ghex/transport_layer/util/barrier.hpp>
#include <ghex/arch_list.hpp>
#include <ghex/unstructured/user_concepts.hpp>
#include <ghex/pattern.hpp>
#include <ghex/unstructured/pattern.hpp>
#include <ghex/unstructured/grid.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/unstructured/communication_object_ipr.hpp>
#include <ghex/common/timer.hpp>
#include <ghex/common/accumulator.hpp>
#include <ghex/common/defs.hpp>
#ifdef GHEX_CUDACC
#include <ghex/common/cuda_runtime.hpp>
#include <ghex/arch_traits.hpp>
#include <ghex/allocator/cuda_allocator.hpp>
#include <ghex/cuda_utils/error.hpp>
#endif


// GHEX type definitions
#ifndef GHEX_TEST_USE_UCX
using transport = gridtools::ghex::tl::mpi_tag;
#else
using transport = gridtools::ghex::tl::ucx_tag;
#endif
using domain_id_type = int;
using global_index_type = idx_t;
using domain_descriptor_type = gridtools::ghex::unstructured::domain_descriptor<domain_id_type, global_index_type>;
using halo_generator_type = gridtools::ghex::unstructured::halo_generator<domain_id_type, global_index_type>;
using grid_type = gridtools::ghex::unstructured::grid;
template<typename T>
using data_descriptor_cpu_type = gridtools::ghex::unstructured::data_descriptor<gridtools::ghex::cpu, domain_id_type, global_index_type, T>;
using timer_type = gridtools::ghex::timer;
#ifdef GHEX_CUDACC
template<typename T>
using gpu_allocator_type = gridtools::ghex::allocator::cuda::allocator<T>;
template<typename T>
using data_descriptor_gpu_type = gridtools::ghex::unstructured::data_descriptor<gridtools::ghex::gpu, domain_id_type, global_index_type, T>;
using device_id_type = gridtools::ghex::arch_traits<gridtools::ghex::gpu>::device_id_type;
#endif


template<typename T>
char* as_bytes(T& i) {
    return reinterpret_cast<char*>(&i);
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
void initialize_field(const Domain& d, Field& f, const O d_id_offset) {
    using value_type = typename Field::value_type;
    assert(f.size() == d.size() * d.levels());
    for (std::size_t i = 0; i < d.inner_size(); ++i) {
        value_type val = static_cast<value_type>(d.domain_id()) * d_id_offset + static_cast<value_type>(d.vertices()[i]);
        for (std::size_t level = 0; level < d.levels(); ++level) {
            f[i * d.levels() + level] = val; // TO DO: use different values for different levels
        }
    }
}

template<typename Domain, typename Pattern, typename Field, typename O>
void check_exchanged_data(const Domain& d, const Pattern& p, const Field& f, const O d_id_offset) {
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
        value_type expected = static_cast<value_type>(pair.second) * d_id_offset + static_cast<value_type>(d.vertices()[pair.first]);
        for (std::size_t level = 0; level < d.levels(); ++level) {
            EXPECT_EQ(f[pair.first * d.levels() + level], expected);
        }
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
    Domain res{d.domain_id(), vs, d.inner_size(), d.levels()};
    return res;
}

template<typename DomainId>
int domain_to_rank(const DomainId d_id, const int num_threads) {
    return d_id / num_threads;
}

template<typename DomainId>
std::vector<DomainId> rank_to_domains(const int rank, const int num_threads) {
    std::vector<DomainId> res(num_threads);
    for (int i = 0; i < num_threads; ++i) {
        res[i] = rank * num_threads + i;
    }
    return res;
}

template<typename DomainId, typename VertexId>
struct d_v_pair {

    using domain_id_type = DomainId;
    using v_id_type = VertexId;

    domain_id_type d_id;
    v_id_type v_id;

    /** @brief unique ordering given by domain id and vertex id*/
    bool operator < (const d_v_pair& other) const noexcept {
        return d_id < other.d_id ? true : (d_id == other.d_id ? v_id < other.v_id : false);
    }

};

using vertices_dist_type = std::map<int, std::map<d_v_pair<domain_id_type, idx_t>, std::vector<idx_t>>>;
using domain_vertices_dist_type = std::map<domain_id_type, std::map<idx_t, std::vector<idx_t>>>;
domain_vertices_dist_type distribute_parmetis(vertices_dist_type& vertices_dist, std::size_t n_vertices, MPI_Comm comm) {
    
    int size;
    MPI_Comm_size(comm, &size);

    // 1) all-to-all: number of vertices per rank
    std::vector<int> s_n_vertices_rank(size);
    for (int i = 0; i < size; ++i) {
        s_n_vertices_rank[i] = vertices_dist[i].size(); // any missing rank gets actually inserted into the map here
    };
    std::vector<int> r_n_vertices_rank(size);
    MPI_Alltoall(s_n_vertices_rank.data(), sizeof(int), MPI_BYTE,
                 r_n_vertices_rank.data(), sizeof(int), MPI_BYTE,
                 comm);

    // 2) all-to-all: vertex ids
    std::vector<idx_t> s_v_ids_rank{};
    s_v_ids_rank.reserve(n_vertices);
    for (const auto& r_m_pair : vertices_dist) {
        for (const auto& v_a_pair : r_m_pair.second) {
            s_v_ids_rank.push_back(v_a_pair.first.v_id);
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

    // 3) all-to-all: domain ids
    std::vector<domain_id_type> s_d_ids_rank{};
    s_d_ids_rank.reserve(n_vertices);
    for (const auto& r_m_pair : vertices_dist) {
        for (const auto& v_a_pair : r_m_pair.second) {
            s_d_ids_rank.push_back(v_a_pair.first.d_id);
        }
    }
    std::vector<int> s_d_ids_rank_counts = counts_as_bytes<domain_id_type>(s_n_vertices_rank);
    std::vector<int> s_d_ids_rank_displs = counts_to_displs(s_d_ids_rank_counts);
    std::vector<domain_id_type> r_d_ids_rank(std::accumulate(r_n_vertices_rank.begin(), r_n_vertices_rank.end(), 0));
    std::vector<int> r_d_ids_rank_counts = counts_as_bytes<domain_id_type>(r_n_vertices_rank);
    std::vector<int> r_d_ids_rank_displs = counts_to_displs(r_d_ids_rank_counts);
    MPI_Alltoallv(s_d_ids_rank.data(), s_d_ids_rank_counts.data(), s_d_ids_rank_displs.data(), MPI_BYTE,
                  r_d_ids_rank.data(), r_d_ids_rank_counts.data(), r_d_ids_rank_displs.data(), MPI_BYTE,
                  comm);

    // 4) all-to-all: adjacency size per vertex per rank
    std::vector<int> s_adjncy_size_vertex_rank{};
    s_adjncy_size_vertex_rank.reserve(n_vertices);
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

    // 6) per-domain vertices distribution map
    domain_vertices_dist_type domain_vertices_dist{};
    for (std::size_t i = 0, a_idx = 0; i < r_v_ids_rank.size(); ++i) {
        auto a_begin = r_adjncy_rank.begin() + a_idx;
        auto a_end = a_begin + r_adjncy_size_vertex_rank[i];
        domain_vertices_dist[r_d_ids_rank[i]]
                .insert(std::make_pair(r_v_ids_rank[i], std::vector<idx_t>{a_begin, a_end}));
        a_idx += r_adjncy_size_vertex_rank[i];
    }

    return domain_vertices_dist;

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

    // Threads
    auto env_threads = std::getenv("GHEX_PARMETIS_BENCHMARK_NUM_THREADS");
    int num_threads = (env_threads) ? std::atoi(env_threads) : 1;

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
    idx_t ncon = 1; // TO DO: might vary
    idx_t nparts = size * num_threads;
    std::vector<real_t> tpwgts_v(ncon * nparts, 1 / static_cast<real_t>(nparts)); // TO DO: might vary
    std::vector<real_t> ubvec_v(ncon, 1.02); // TO DO: might vary
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
    vertices_dist_type vertices_dist{};
    for (idx_t v_id = vtxdist_v[rank], i = 0; i < static_cast<idx_t>(ap_n.size() - 1); ++v_id, ++i) {
        vertices_dist[domain_to_rank(part_v[i], num_threads)]
                .insert(std::make_pair(d_v_pair<domain_id_type, idx_t>{static_cast<domain_id_type>(part_v[i]), v_id}, std::vector<idx_t>{ai.begin() + ap_n[i], ai.begin() + ap_n[i+1]}));
    }
    auto domain_vertices_dist = distribute_parmetis(vertices_dist, ap_n.size() - 1, comm);

    // GHEX constants
    const std::size_t levels = 100;
    const idx_t d_id_offset = 10e9;
    const int n_iters_warm_up = 50;
    const int n_iters = 50;

#ifndef GHEX_CUDACC

    // GHEX context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int gh_rank = context.rank();

    // barrier
    gridtools::ghex::tl::barrier_t gh_barrier{static_cast<std::size_t>(num_threads)};

    // timers (local = rank local)
#ifdef GHEX_PARMETIS_BENCHMARK_UNORDERED
    timer_type t_buf_local; // 1 - unordered halos - buffered receive
    std::mutex t_buf_local_mutex;
#endif
#ifdef GHEX_PARMETIS_BENCHMARK_ORDERED
    timer_type t_ord_buf_local; // 2 - ordered halos - buffered receive
    std::mutex t_ord_buf_local_mutex;
#endif
#ifdef GHEX_PARMETIS_BENCHMARK_IPR
    timer_type t_ord_ipr_local; // 3 - ordered halos - in-place receive
    std::mutex t_ord_ipr_local_mutex;
#endif

    // output file
    std::stringstream ss_file;
    ss_file << gh_rank;
    std::string filename =
#ifdef GHEX_PARMETIS_BENCHMARK_UNORDERED
            "unstructured_parmetis_receive_type_unordered_"
#endif
#ifdef GHEX_PARMETIS_BENCHMARK_ORDERED
            "unstructured_parmetis_receive_type_ordered_"
#endif
#ifdef GHEX_PARMETIS_BENCHMARK_IPR
            "unstructured_parmetis_receive_type_ipr_"
#endif
            + ss_file.str() + ".txt";
    std::ofstream file(filename.c_str());
    file << "Unstructured ParMETIS receive type benchmark\n\n";

    // 1 ======== unordered halos - buffered receive =========================

    // setup
    std::vector<domain_descriptor_type> local_domains{};
    for (auto d_id : rank_to_domains<domain_id_type>(gh_rank, num_threads)) {
        std::vector<idx_t> vertices{};
        vertices.reserve(domain_vertices_dist[d_id].size()); // any missing domain gets actually inserted into the map here
        std::vector<idx_t> adjncy{}; // size may be computed in advance, not preformance critical anyway
        for (const auto& v_a_pair : domain_vertices_dist[d_id]) {
            vertices.push_back(v_a_pair.first);
            adjncy.insert(adjncy.end(), v_a_pair.second.begin(), v_a_pair.second.end());
        }
        local_domains.push_back(domain_descriptor_type{d_id, vertices, adjncy, levels}); // CSR constructor
    }
    halo_generator_type hg{};
    auto p = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);
    using pattern_container_type = decltype(p);

#ifdef GHEX_PARMETIS_BENCHMARK_UNORDERED

    std::vector<std::vector<idx_t>> f{};
    std::vector<data_descriptor_cpu_type<idx_t>> data{};
    for (const auto& d : local_domains) {
        std::vector<idx_t> local_f(d.size() * d.levels(), 0);
        initialize_field(d, local_f, d_id_offset);
        f.push_back(std::move(local_f));
        data.push_back(data_descriptor_cpu_type<idx_t>{d, f.back()});
    }

    // thread function
    auto thread_func = [&context, &gh_barrier, &t_buf_local, &t_buf_local_mutex](auto bi){
        auto th_comm = context.get_communicator();
        timer_type t_buf_local_th;
        auto co = gridtools::ghex::make_communication_object<pattern_container_type>(th_comm);
        for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
            auto h = co.exchange(bi);
            h.wait();
        }
        for (int i = 0; i < n_iters; ++i) { // benchmark
            timer_type t_local;
            gh_barrier(th_comm);
            t_local.tic();
            auto h = co.exchange(bi);
            h.wait();
            t_local.toc();
            t_buf_local_th(t_local);
        }
        std::lock_guard<std::mutex> guard(t_buf_local_mutex);
        t_buf_local(t_buf_local_th);
    };

    // run
    std::vector<std::thread> threads{};
    for (auto& d : data) {
        threads.push_back(std::thread{thread_func, p(d)});
    }
    for (auto& t : threads) t.join();

    // check
    for (std::size_t i = 0; i < f.size(); ++i) {
        check_exchanged_data(local_domains[i], p[i], f[i], d_id_offset);
    }

    // global time
    auto t_buf_global = gridtools::ghex::reduce(t_buf_local, context.mpi_comm());

    // exchanged size
    idx_t n_halo_vertices_local{0}, n_halo_vertices_global;
    for (const auto& d : local_domains) {
        n_halo_vertices_local += (d.size() - d.inner_size());
    }
    MPI_Allreduce(&n_halo_vertices_local, &n_halo_vertices_global, 1, MPI_INT64_T, MPI_SUM, context.mpi_comm()); // MPI type set according to parmetis idx type

    // output
    file << "total exchanged size in GB (assuming value type = idx_t): "
         << static_cast<double>(n_halo_vertices_global * levels * sizeof(idx_t) * 2) / (1024.0 * 1024.0 * 1024.0) << "\n\n"
         << "1 - unordered halos - buffered receive - CPU\n"
         << "\tlocal time = " << t_buf_local.mean() / 1000.0
         << "+/-" << t_buf_local.stddev() / (std::sqrt(t_buf_local.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_buf_global.mean() / 1000.0
         << "+/-" << t_buf_global.stddev() / (std::sqrt(t_buf_global.num_samples()) * 1000.0) << "ms\n";

#endif

    // 2 ======== ordered halos - buffered receive ===========================

    // setup
    std::vector<domain_descriptor_type> local_domains_ord{};
    for (std::size_t i = 0; i < local_domains.size(); ++i) {
        local_domains_ord.push_back(make_reindexed_domain(local_domains[i], p[i]));
    }
    auto p_ord = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains_ord); // easiest way, but quite redundant: only recv halos are different

#ifdef GHEX_PARMETIS_BENCHMARK_ORDERED

    std::vector<std::vector<idx_t>> f_ord{};
    std::vector<data_descriptor_cpu_type<idx_t>> data_ord{};
    for (const auto& d_ord : local_domains_ord) {
        std::vector<idx_t> local_f_ord(d_ord.size() * d_ord.levels(), 0);
        initialize_field(d_ord, local_f_ord, d_id_offset);
        f_ord.push_back(std::move(local_f_ord));
        data_ord.push_back(data_descriptor_cpu_type<idx_t>{d_ord, f_ord.back()});
    }

    // thread function
    auto thread_func_ord = [&context, &gh_barrier, &t_ord_buf_local, &t_ord_buf_local_mutex](auto bi){
        auto th_comm = context.get_communicator();
        timer_type t_ord_buf_local_th;
        auto co_ord = gridtools::ghex::make_communication_object<pattern_container_type>(th_comm);
        for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
            auto h_ord = co_ord.exchange(bi);
            h_ord.wait();
        }
        for (int i = 0; i < n_iters; ++i) { // benchmark
            timer_type t_local;
            gh_barrier(th_comm);
            t_local.tic();
            auto h_ord = co_ord.exchange(bi);
            h_ord.wait();
            t_local.toc();
            t_ord_buf_local_th(t_local);
        }
        std::lock_guard<std::mutex> guard(t_ord_buf_local_mutex);
        t_ord_buf_local(t_ord_buf_local_th);
    };

    // run
    std::vector<std::thread> threads_ord{};
    for (auto& d_ord : data_ord) {
        threads_ord.push_back(std::thread{thread_func_ord, p_ord(d_ord)});
    }
    for (auto& t_ord : threads_ord) t_ord.join();

    // check
    for (std::size_t i = 0; i < f_ord.size(); ++i) {
        check_exchanged_data(local_domains_ord[i], p_ord[i], f_ord[i], d_id_offset);
    }

    // global time
    auto t_ord_buf_global = gridtools::ghex::reduce(t_ord_buf_local, context.mpi_comm());

    file << "2 - ordered halos - buffered receive - CPU\n"
         << "\tlocal time = " << t_ord_buf_local.mean() / 1000.0
         << "+/-" << t_ord_buf_local.stddev() / (std::sqrt(t_ord_buf_local.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_ord_buf_global.mean() / 1000.0
         << "+/-" << t_ord_buf_global.stddev() / (std::sqrt(t_ord_buf_global.num_samples()) * 1000.0) << "ms\n";

#endif

    // 3 ======== ordered halos - in-place receive ===========================

#ifdef GHEX_PARMETIS_BENCHMARK_IPR

    std::vector<std::vector<idx_t>> f_ipr{};
    std::vector<data_descriptor_cpu_type<idx_t>> data_ipr{};
    for (const auto& d_ord : local_domains_ord) {
        std::vector<idx_t> local_f_ipr(d_ord.size() * d_ord.levels(), 0);
        initialize_field(d_ord, local_f_ipr, d_id_offset);
        f_ipr.push_back(std::move(local_f_ipr));
        data_ipr.push_back(data_descriptor_cpu_type<idx_t>{d_ord, f_ipr.back()});
    }

    // thread function
    auto thread_func_ipr = [&context, &gh_barrier, &t_ord_ipr_local, &t_ord_ipr_local_mutex](auto bi){
        auto th_comm = context.get_communicator();
        timer_type t_ord_ipr_local_th;
        auto co_ipr = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(th_comm);
        for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
            auto h_ipr = co_ipr.exchange(bi);
            h_ipr.wait();
        }
        for (int i = 0; i < n_iters; ++i) { // benchmark
            timer_type t_local;
            gh_barrier(th_comm);
            t_local.tic();
            auto h_ipr = co_ipr.exchange(bi);
            h_ipr.wait();
            t_local.toc();
            t_ord_ipr_local_th(t_local);
        }
        std::lock_guard<std::mutex> guard(t_ord_ipr_local_mutex);
        t_ord_ipr_local(t_ord_ipr_local_th);
    };

    // run
    std::vector<std::thread> threads_ipr{};
    for (auto& d_ipr : data_ipr) {
        threads_ipr.push_back(std::thread{thread_func_ipr, p_ord(d_ipr)});
    }
    for (auto& t_ipr : threads_ipr) t_ipr.join();

    // check
    for (std::size_t i = 0; i < f_ipr.size(); ++i) {
        check_exchanged_data(local_domains_ord[i], p_ord[i], f_ipr[i], d_id_offset);
    }

    // global time
    auto t_ord_ipr_global = gridtools::ghex::reduce(t_ord_ipr_local, context.mpi_comm());

    file << "3 - ordered halos - in-place receive - CPU\n"
         << "\tlocal time = " << t_ord_ipr_local.mean() / 1000.0
         << "+/-" << t_ord_ipr_local.stddev() / (std::sqrt(t_ord_ipr_local.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_ord_ipr_global.mean() / 1000.0
         << "+/-" << t_ord_ipr_global.stddev() / (std::sqrt(t_ord_ipr_global.num_samples()) * 1000.0) << "ms\n";

#endif

#else

    // GHEX context
    auto context_ptr = gridtools::ghex::tl::context_factory<transport>::create(MPI_COMM_WORLD);
    auto& context = *context_ptr;
    int gh_rank = context.rank();
    auto gh_comm = context.get_communicator();
    int num_devices;
    GHEX_CHECK_CUDA_RESULT(cudaGetDeviceCount(&num_devices));
    device_id_type device_id = gh_rank % num_devices;
    GHEX_CHECK_CUDA_RESULT(cudaSetDevice(device_id));

    // timers
    timer_type t_buf_local_gpu, t_buf_global_gpu; // 1 - unordered halos - buffered receive
    timer_type t_ord_buf_local_gpu, t_ord_buf_global_gpu; // 2 - ordered halos - buffered receive
    timer_type t_ord_ipr_local_gpu, t_ord_ipr_global_gpu; // 3 - ordered halos - in-place receive

    // output file
    std::stringstream ss_file;
    ss_file << gh_rank;
    std::string filename = "unstructured_parmetis_receive_type_gpu_" + ss_file.str() + ".txt";
    std::ofstream file(filename.c_str());
    file << "Unstructured ParMETIS receive type benchmark; DEBUG: GPU device id = " << device_id << "\n\n";

    // GPU allocator
    gpu_allocator_type<idx_t> gpu_alloc{};

    // 1 ======== unordered halos - buffered receive =========================

    // setup
    domain_id_type d_id{gh_rank}; // 1 domain per rank
    std::vector<idx_t> vertices{};
    vertices.reserve(domain_vertices_dist[d_id].size()); // any missing domain gets actually inserted into the map here
    std::vector<idx_t> adjncy{}; // size may be computed in advance, not preformance critical anyway
    for (const auto& v_a_pair : domain_vertices_dist[d_id]) {
	vertices.push_back(v_a_pair.first);
        adjncy.insert(adjncy.end(), v_a_pair.second.begin(), v_a_pair.second.end());
    }
    domain_descriptor_type d{d_id, vertices, adjncy, levels}; // CSR constructor
    std::vector<domain_descriptor_type> local_domains{d};
    halo_generator_type hg{};
    auto p = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains);
    using pattern_container_type = decltype(p);
    auto co = gridtools::ghex::make_communication_object<pattern_container_type>(gh_comm);
    std::vector<idx_t> f_cpu(d.size() * d.levels(), 0);
    initialize_field(d, f_cpu, d_id_offset);
    idx_t* f_gpu = gpu_alloc.allocate(d.size() * d.levels());
    GHEX_CHECK_CUDA_RESULT(cudaMemcpy(f_gpu, f_cpu.data(), d.size() * d.levels() * sizeof(idx_t), cudaMemcpyHostToDevice));
    data_descriptor_gpu_type<idx_t> data_gpu{d, f_gpu, 1, true, device_id};

    // exchange
    for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
        auto h_gpu = co.exchange(p(data_gpu));
        h_gpu.wait();
    }
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_gpu = co.exchange(p(data_gpu));
        h_gpu.wait();
        t_local.toc();
        t_buf_local_gpu(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_buf_global_gpu(t_global);
    }

    // check
    cudaMemcpy(f_cpu.data(), f_gpu, d.size() * d.levels() * sizeof(idx_t), cudaMemcpyDeviceToHost);
    check_exchanged_data(d, p[0], f_cpu, d_id_offset);

    // deallocate
    gpu_alloc.deallocate(f_gpu, d.size() * d.levels());

    // 2 ======== ordered halos - buffered receive ===========================

    // setup
    domain_descriptor_type d_ord = make_reindexed_domain(d, p[0]);
    std::vector<domain_descriptor_type> local_domains_ord{d_ord};
    auto p_ord = gridtools::ghex::make_pattern<grid_type>(context, hg, local_domains_ord); // easiest way, but quite redundant: only recv halos are different
    auto co_ord = gridtools::ghex::make_communication_object<pattern_container_type>(gh_comm); // new one, same conditions
    std::vector<idx_t> f_ord_cpu(d_ord.size() * d_ord.levels(), 0);
    initialize_field(d_ord, f_ord_cpu, d_id_offset);
    idx_t* f_ord_gpu = gpu_alloc.allocate(d_ord.size() * d_ord.levels());
    GHEX_CHECK_CUDA_RESULT(cudaMemcpy(f_ord_gpu, f_ord_cpu.data(), d_ord.size() * d_ord.levels() * sizeof(idx_t), cudaMemcpyHostToDevice));
    data_descriptor_gpu_type<idx_t> data_ord_gpu{d_ord, f_ord_gpu, 1, true, device_id};

    // exchange
    for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
        auto h_ord_gpu = co_ord.exchange(p_ord(data_ord_gpu));
        h_ord_gpu.wait();
    }
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_ord_gpu = co_ord.exchange(p_ord(data_ord_gpu));
        h_ord_gpu.wait();
        t_local.toc();
        t_ord_buf_local_gpu(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ord_buf_global_gpu(t_global);
    }

    // check
    cudaMemcpy(f_ord_cpu.data(), f_ord_gpu, d_ord.size() * d_ord.levels() * sizeof(idx_t), cudaMemcpyDeviceToHost);
    check_exchanged_data(d_ord, p_ord[0], f_ord_cpu, d_id_offset);

    // deallocate
    gpu_alloc.deallocate(f_ord_gpu, d_ord.size() * d_ord.levels());

    // 3 ======== ordered halos - in-place receive ===========================

    // setup
    auto co_ipr = gridtools::ghex::make_communication_object_ipr<pattern_container_type>(gh_comm);
    std::vector<idx_t> f_ipr_cpu(d_ord.size() * d_ord.levels(), 0);
    initialize_field(d_ord, f_ipr_cpu, d_id_offset);
    idx_t* f_ipr_gpu = gpu_alloc.allocate(d_ord.size() * d_ord.levels());
    GHEX_CHECK_CUDA_RESULT(cudaMemcpy(f_ipr_gpu, f_ipr_cpu.data(), d_ord.size() * d_ord.levels() * sizeof(idx_t), cudaMemcpyHostToDevice));
    data_descriptor_gpu_type<idx_t> data_ipr_gpu{d_ord, f_ipr_gpu, 1, true, device_id};

    // exchange
    for (int i = 0; i < n_iters_warm_up; ++i) { // warm-up
        auto h_ipr_gpu = co_ipr.exchange(p_ord(data_ipr_gpu));
        h_ipr_gpu.wait();
    }
    for (int i = 0; i < n_iters; ++i) { // benchmark
        timer_type t_local;
        MPI_Barrier(context.mpi_comm());
        t_local.tic();
        auto h_ipr_gpu = co_ipr.exchange(p_ord(data_ipr_gpu));
        h_ipr_gpu.wait();
        t_local.toc();
        t_ord_ipr_local_gpu(t_local);
        MPI_Barrier(context.mpi_comm());
        auto t_global = gridtools::ghex::reduce(t_local, context.mpi_comm());
        t_ord_ipr_global_gpu(t_global);
    }

    // check
    cudaMemcpy(f_ipr_cpu.data(), f_ipr_gpu, d_ord.size() * d_ord.levels() * sizeof(idx_t), cudaMemcpyDeviceToHost);
    check_exchanged_data(d_ord, p_ord[0], f_ipr_cpu, d_id_offset);

    // deallocate
    gpu_alloc.deallocate(f_ipr_gpu, d_ord.size() * d_ord.levels());

    // ======== output =======================================================

    idx_t n_halo_vertices_local{static_cast<idx_t>(d.size() - d.inner_size())}, n_halo_vertices_global;
    MPI_Allreduce(&n_halo_vertices_local, &n_halo_vertices_global, 1, MPI_INT64_T, MPI_SUM, context.mpi_comm()); // MPI type set according to parmetis idx type
    file << "total exchanged size in GB (assuming value type = idx_t): "
         << static_cast<double>(n_halo_vertices_global * levels * sizeof(idx_t) * 2) / (1024.0 * 1024.0 * 1024.0) << "\n\n";

    file << "1 - unordered halos - buffered receive - GPU\n"
         << "\tlocal time = " << t_buf_local_gpu.mean() / 1000.0
         << "+/-" << t_buf_local_gpu.stddev() / (std::sqrt(t_buf_local_gpu.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_buf_global_gpu.mean() / 1000.0
         << "+/-" << t_buf_global_gpu.stddev() / (std::sqrt(t_buf_global_gpu.num_samples()) * 1000.0) << "ms\n";

    file << "2 - ordered halos - buffered receive - GPU\n"
         << "\tlocal time = " << t_ord_buf_local_gpu.mean() / 1000.0
         << "+/-" << t_ord_buf_local_gpu.stddev() / (std::sqrt(t_ord_buf_local_gpu.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_ord_buf_global_gpu.mean() / 1000.0
         << "+/-" << t_ord_buf_global_gpu.stddev() / (std::sqrt(t_ord_buf_global_gpu.num_samples()) * 1000.0) << "ms\n";

    file << "3 - ordered halos - in-place receive - GPU\n"
         << "\tlocal time = " << t_ord_ipr_local_gpu.mean() / 1000.0
         << "+/-" << t_ord_ipr_local_gpu.stddev() / (std::sqrt(t_ord_ipr_local_gpu.num_samples()) * 1000.0) << "ms\n"
         << "\tglobal time = " << t_ord_ipr_global_gpu.mean() / 1000.0
         << "+/-" << t_ord_ipr_global_gpu.stddev() / (std::sqrt(t_ord_ipr_global_gpu.num_samples()) * 1000.0) << "ms\n";

#endif

    // MPI setup
    MPI_Comm_free(&comm);

}
