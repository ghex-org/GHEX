/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <gtest/gtest.h>
#include "../../mpi_runner/mpi_test_fixture.hpp"

#include <ghex/config.hpp>
#include <ghex/bulk_communication_object.hpp>
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/rma_range_generator.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/structured/regular/make_pattern.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include "../../util/memory.hpp"
#include <gridtools/common/array.hpp>
#include <array>
#include <iostream>
#include <vector>
#include <future>

using namespace ghex;
using arr = std::array<int, 2>;
using domain = structured::regular::domain_descriptor<int, std::integral_constant<int, 2>>;
using halo_gen = structured::regular::halo_generator<int, std::integral_constant<int, 2>>;

#define DIM  8
#define HALO 3
#define PX   true
#define PY   true

constexpr std::array<int, 4>  halos{HALO, HALO, HALO, HALO};
constexpr std::array<bool, 2> periodic{PX, PY};

ghex::test::util::memory<gridtools::array<int, 2>>
allocate_field()
{
    return {(HALO * 2 + DIM) * (HALO * 2 + DIM / 2), gridtools::array<int, 2>{-1, -1}};
}

template<typename RawField>
auto
wrap_cpu_field(RawField& raw_field, const domain& d)
{
    return wrap_field<cpu, ::gridtools::layout_map<1, 0>>(
        d, raw_field.data(), arr{HALO, HALO}, arr{HALO * 2 + DIM, HALO * 2 + DIM / 2});
}

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
template<typename RawField>
auto
wrap_gpu_field(RawField& raw_field, const domain& d)
{
    return wrap_field<gpu, ::gridtools::layout_map<1, 0>>(
        d, raw_field.device_data(), arr{HALO, HALO}, arr{HALO * 2 + DIM, HALO * 2 + DIM / 2});
}
#endif

template<typename Field>
Field&&
fill(Field&& field)
{
    for (int j = 0; j < DIM / 2; ++j)
        for (int i = 0; i < DIM; ++i)
        {
            auto& v = field({i, j});
            v[0] = field.domain().first()[0] + i;
            v[1] = field.domain().first()[1] + j;
        }
    return std::forward<Field>(field);
}

template<typename Field>
Field&&
reset(Field&& field)
{
    for (int j = -HALO; j < DIM / 2 + HALO; ++j)
        for (int i = -HALO; i < DIM + HALO; ++i)
        {
            auto& v = field({i, j});
            v[0] = -1;
            v[1] = -1;
        }
    return fill(std::forward<Field>(field));
}

int
expected(int coord, int dim, int first, int last, bool periodic_)
{
    const auto ext = (last + 1) - first;
    int        res = first + coord;
    if (first == 0 && coord < 0) res = periodic_ ? dim * DIM + coord : -1;
    if (last == dim * DIM - 1 && coord >= ext) res = periodic_ ? coord - ext : -1;
    return res;
}

template<typename Arr>
bool
compare(const Arr& v, int x, int y)
{
    if (x == -1 || y == -1)
    {
        x = -1;
        y = -1;
    }
    return (x == v[0]) && (y == v[1]);
}

template<typename Field>
bool
check(const Field& field, const arr& dims)
{
    bool res = true;
    for (int j = -HALO; j < DIM / 2 + HALO; ++j)
    {
        const auto y =
            expected(j, dims[1], field.domain().first()[1], field.domain().last()[1], periodic[1]);
        for (int i = -HALO; i < DIM + HALO; ++i)
        {
            const auto x = expected(
                i, dims[0], field.domain().first()[0], field.domain().last()[0], periodic[0]);
            res = res && compare(field({i, j}), x, y);
        }
    }
    return res;
}

auto
make_domain(int rank, int id, std::array<int, 2> coord)
{
    const auto x = coord[0] * DIM;
    const auto y = (coord[1] * 2 + id) * (DIM / 2);
    return domain{rank * 2 + id, arr{x, y}, arr{x + DIM - 1, y + DIM / 2 - 1}};
}

struct domain_lu
{
    struct neighbor
    {
        int m_rank;
        int m_id;
        int rank() const noexcept { return m_rank; }
        int id() const noexcept { return m_id; }
    };

    arr m_dims;

    neighbor operator()(int id, arr const& offset) const noexcept
    {
        auto rank = id / 2;
        auto y_ = id - 2 * rank;
        auto y = rank / m_dims[0];
        auto x = rank - y * m_dims[0];
        y = ((2 * y + y_ + offset[1]) + m_dims[1] * 2) % (m_dims[1] * 2);
        x = (x + offset[0] + m_dims[0]) % m_dims[0];

        int n_rank = (y / 2) * m_dims[0] + x;
        int n_id = 2 * n_rank + (y % 2);
        return {n_rank, n_id};
    }
};

template<typename Pattern, typename SPattern, typename Domains>
bool
run(context& ctxt, const Pattern& pattern, const SPattern& spattern, const Domains& domains,
    const arr& dims, int thread_id)
{
    bool res = true;
    // field
    auto raw_field = allocate_field();
    auto field = fill(wrap_cpu_field(raw_field, domains[thread_id]));
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_device();
    auto field_gpu = wrap_gpu_field(raw_field, domains[thread_id]);
#endif

    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(ctxt);

    // classical
    // ---------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id == 0) co.exchange(pattern(field)).wait();
    else
        co.exchange(pattern(field_gpu)).wait();
#else
    co.exchange(pattern(field)).wait();
#endif

        // check field
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_device();
#endif

        //barrier(comm);

        // using stages
        // ------------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id == 0)
    {
        co.exchange(spattern[0]->operator()(field)).wait();
        co.exchange(spattern[1]->operator()(field)).wait();
    }
    else
    {
        co.exchange(spattern[0]->operator()(field_gpu)).wait();
        co.exchange(spattern[1]->operator()(field_gpu)).wait();
    }
#else
    co.exchange(spattern[0]->operator()(field)).wait();
    co.exchange(spattern[1]->operator()(field)).wait();
#endif

    // check field
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_device();
#endif

        //barrier(comm);

        // bulk exchange (rma)
        // ===================

        // classical
        // ---------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field),
        decltype(field_gpu)>(ctxt);
    if (thread_id == 0) bco.add_field(pattern(field));
    else
        bco.add_field(pattern(field_gpu));
#else
    auto bco =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(ctxt);
    bco.add_field(pattern(field));
#endif
    //bco.exchange().wait();
    generic_bulk_communication_object gbco(std::move(bco));
    gbco.exchange().wait();

    // check field
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_host();
#endif
    res = res && check(field, dims);

    // reset field
    reset(field);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_device();
#endif

        //barrier(comm);

        // using stages
        // ------------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern,
        decltype(field), decltype(field_gpu)>(ctxt);
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern,
        decltype(field), decltype(field_gpu)>(ctxt);
    if (thread_id == 0)
    {
        bco_x.add_field(spattern[0]->operator()(field));
        bco_y.add_field(spattern[1]->operator()(field));
    }
    else
    {
        bco_x.add_field(spattern[0]->operator()(field_gpu));
        bco_y.add_field(spattern[1]->operator()(field_gpu));
    }
#else
    auto bco_x =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(ctxt);
    auto bco_y =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field)>(ctxt);
    bco_x.add_field(spattern[0]->operator()(field));
    bco_y.add_field(spattern[1]->operator()(field));
#endif
    generic_bulk_communication_object gbco_x(std::move(bco_x));
    generic_bulk_communication_object gbco_y(std::move(bco_y));
    gbco_x.exchange().wait();
    gbco_y.exchange().wait();

    // check field
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    if (thread_id != 0) raw_field.clone_to_host();
#endif
    res = res && check(field, dims);
    return res;
}

template<typename Pattern, typename SPattern, typename Domains>
bool
run(context& ctxt, const Pattern& pattern, const SPattern& spattern, const Domains& domains,
    const arr& dims)
{
    bool res = true;
    // fields
    auto raw_field_a = allocate_field();
    auto raw_field_b = allocate_field();
    auto field_a = fill(wrap_cpu_field(raw_field_a, domains[0]));
    auto field_b = fill(wrap_cpu_field(raw_field_b, domains[1]));
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_device();
    auto field_b_gpu = wrap_gpu_field(raw_field_b, domains[1]);
#endif

    // general exchange
    // ================
    auto co = make_communication_object<Pattern>(ctxt);

    // classical
    // ---------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    co.exchange(pattern(field_a), pattern(field_b_gpu)).wait();
#else
    co.exchange(pattern(field_a), pattern(field_b)).wait();
#endif

    // check fields
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_device();
#endif

    //barrier(comm);

    // using stages
    // ------------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    co.exchange(spattern[0]->operator()(field_a), spattern[0]->operator()(field_b_gpu)).wait();
    co.exchange(spattern[1]->operator()(field_a), spattern[1]->operator()(field_b_gpu)).wait();
#else
    co.exchange(spattern[0]->operator()(field_a), spattern[0]->operator()(field_b)).wait();
    co.exchange(spattern[1]->operator()(field_a), spattern[1]->operator()(field_b)).wait();
#endif

    // check fields
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_device();
#endif

    //barrier(comm);

    // bulk exchange (rma)
    // ===================

    // classical
    // ---------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    auto bco = bulk_communication_object<structured::rma_range_generator, Pattern,
        decltype(field_a), decltype(field_b_gpu)>(ctxt);
    bco.add_field(pattern(field_a));
    bco.add_field(pattern(field_b_gpu));
#else
    auto bco =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(
            ctxt);
    bco.add_field(pattern(field_a));
    bco.add_field(pattern(field_b));
#endif
    bco.init();
    //barrier(comm);
    bco.exchange().wait();

    // check fields
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    // reset fields
    reset(field_a);
    reset(field_b);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_device();
#endif

    //barrier(comm);

    // using stages
    // ------------

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    auto bco_x = bulk_communication_object<structured::rma_range_generator, Pattern,
        decltype(field_a), decltype(field_b_gpu)>(ctxt);
    bco_x.add_field(spattern[0]->operator()(field_a));
    bco_x.add_field(spattern[0]->operator()(field_b_gpu));
    auto bco_y = bulk_communication_object<structured::rma_range_generator, Pattern,
        decltype(field_a), decltype(field_b_gpu)>(ctxt);
    bco_y.add_field(spattern[1]->operator()(field_a));
    bco_y.add_field(spattern[1]->operator()(field_b_gpu));
#else
    auto bco_x =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(
            ctxt);
    bco_x.add_field(spattern[0]->operator()(field_a));
    bco_x.add_field(spattern[0]->operator()(field_b));
    auto bco_y =
        bulk_communication_object<structured::rma_range_generator, Pattern, decltype(field_a)>(
            ctxt);
    bco_y.add_field(spattern[1]->operator()(field_a));
    bco_y.add_field(spattern[1]->operator()(field_b));
#endif
    bco_x.init();
    bco_y.init();
    //barrier(comm);
    bco_x.exchange().wait();
    bco_y.exchange().wait();

    // check fields
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    raw_field_b.clone_to_host();
#endif
    res = res && check(field_a, dims);
    res = res && check(field_b, dims);

    return res;
}

void
sim(bool multi_threaded)
{
    context ctxt(MPI_COMM_WORLD, multi_threaded);
    // 2D domain decomposition
    arr dims{0, 0}, coords{0, 0};
    MPI_Dims_create(ctxt.size(), 2, dims.data());
    coords[1] = ctxt.rank() / dims[0];
    coords[0] = ctxt.rank() - coords[1] * dims[0];
    // make 2 domains per rank
    std::vector<domain> domains{
        make_domain(ctxt.rank(), 0, coords), make_domain(ctxt.rank(), 1, coords)};
    // neighbor lookup
    domain_lu d_lu{dims};

    auto staged_pattern = structured::regular::make_staged_pattern(
        ctxt, domains, d_lu, arr{0, 0}, arr{dims[0] * DIM - 1, dims[1] * DIM - 1}, halos, periodic);

    // make halo generator
    halo_gen gen{arr{0, 0}, arr{dims[0] * DIM - 1, dims[1] * DIM - 1}, halos, periodic};
    // create a pattern for communication
    auto pattern = make_pattern<structured::grid>(ctxt, gen, domains);
    // run
    bool res = true;
    if (multi_threaded)
    {
        auto run_fct = [&ctxt, &pattern, &staged_pattern, &domains, &dims](int id) {
            return run(ctxt, pattern, staged_pattern, domains, dims, id);
        };
        auto f1 = std::async(std::launch::async, run_fct, 0);
        auto f2 = std::async(std::launch::async, run_fct, 1);
        res = res && f1.get();
        res = res && f2.get();
    }
    else
    {
        res = res && run(ctxt, pattern, staged_pattern, domains, dims);
    }
    // reduce res
    bool all_res = false;
    MPI_Reduce(&res, &all_res, 1, MPI_C_BOOL, MPI_LAND, 0, MPI_COMM_WORLD);
    if (ctxt.rank() == 0) { EXPECT_TRUE(all_res); }
}

TEST_F(mpi_test_fixture, simple_exchange) { sim(thread_safe); }
