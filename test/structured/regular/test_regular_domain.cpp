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
#include <ghex/structured/pattern.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>
#include <ghex/communication_object.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#include <gridtools/common/cuda_util.hpp>
#endif

#include <gridtools/common/array.hpp>
#include "../../util/memory.hpp"
#include <vector>
#include <thread>
#include <future>

// this class sets up the decomposition, fields and patterns for the halo exchange,
// initializes the values before the exchange and can check for correctness of the exchange
template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
struct parameters
{
    // decomposition: 4xYx1
    // - 4 domains in x-direction
    // - 1 domain in z-direction
    // - rest in y-direction
    // - MPI rank owns two sub-domains: either first or last two domains in x-direction
    //
    // architecture/devices:
    // - even subdomains: on architecture Arch_A
    // - odd subdomains: on architecture Arch_B
    // - Arch_A and Arch_B may be equal
    //
    // fields:
    // - 3 fields per sub-domain
    // - fields have different data types (T1, T2, T3)
    // - data types may be equal
    //
    // patterns:
    // - 2 patterns (with different halos)
    // - pattern_1 applies to fields 1 and 3
    // - pattern 2 applies to field 2
    //
    //
    //          +---------> x
    //          |
    //          |     +------<0>------+------<1>------+
    //          |     | +----+ +----+ | +----+ +----+ |
    //          v     | |  0 | |  1 | | |  2 | |  3 | |
    //                | +----+ +----+ | +----+ +----+ |
    //          y     +------<2>------+------<3>------+
    //                | +----+ +----+ | +----+ +----+ |
    //                | |  4 | |  5 | | |  6 | |  7 | |
    //                | +----+ +----+ | +----+ +----+ |
    //                +------<4>------+------<5>------+
    //                | +----+ +----+ | +----+ +----+ |
    //                | |  8 | |  9 | | | 10 | | 11 | |
    //                . .    . .    . . .    . .    . .
    //                . .    . .    . . .    . .    . .
    //

    template<typename T, std::size_t N>
    using array_type = gridtools::array<T, N>;
    using TT1 = array_type<T1, 3>;
    using TT2 = array_type<T2, 3>;
    using TT3 = array_type<T3, 3>;
    using domain_descriptor_type =
        ghex::structured::regular::domain_descriptor<int, std::integral_constant<int, 3>>;
    using halo_generator_type =
        ghex::structured::regular::halo_generator<int, std::integral_constant<int, 3>>;
    template<typename T, typename A>
    using field_type = ghex::structured::regular::field_descriptor<T, A, domain_descriptor_type,
        gridtools::layout_map<2, 1, 0>>;

    template<typename U, typename Offsets, typename Extents>
    static field_type<U, ghex::cpu> wrap(ghex::test::util::memory<U>& f, domain_descriptor_type& d,
        Offsets const& o, Extents const& ext, ghex::cpu)
    {
        return ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
            d, f.host_data(), o, ext);
    }

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    template<typename U, typename Offsets, typename Extents>
    static field_type<U, ghex::gpu> wrap(ghex::test::util::memory<U>& f, domain_descriptor_type& d,
        Offsets const& o, Extents const& ext, ghex::gpu)
    {
        return ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            d, f.device_data(), o, ext);
    }
#endif

    ghex::context& ctxt;
    // local portion per domain
    const std::array<int, 3>  local_ext{4, 3, 2};
    const std::array<bool, 3> periodic{true, true, true};
    // maximum halo
    const std::array<int, 3> offset{3, 3, 3};
    // halos
    const std::array<int, 6> halos_1{0, 0, 1, 0, 1, 2};
    const std::array<int, 6> halos_2{2, 2, 2, 2, 2, 2};
    // compute total domain
    const std::array<int, 3> g_first{0, 0, 0};
    const std::array<int, 3> g_last;
    // local size including potential halos
    const std::array<int, 3> local_ext_buffer;
    // maximum number of elements per local domain
    const int max_memory;
    // local domains
    std::vector<domain_descriptor_type> local_domains;
    // pattern containers
    using pattern_container_type = decltype(ghex::make_pattern<ghex::structured::grid>(
        ctxt, std::declval<halo_generator_type>(), local_domains));
    std::unique_ptr<pattern_container_type> pattern1;
    std::unique_ptr<pattern_container_type> pattern2;

    // fields
    template<typename T, typename Arch>
    struct field
    {
        ghex::test::util::memory<T> raw_field;
        field_type<T, Arch>         ghex_field;
        domain_descriptor_type&     dom;
        std::array<int, 6>          halos;
        pattern_container_type&     pattern;
        ghex::buffer_info<typename pattern_container_type::value_type, Arch, field_type<T, Arch>>
            bi;

        field(int size, std::array<int, 3> const& off, std::array<int, 3> const& ext,
            domain_descriptor_type& d, std::array<int, 6> const halos_, pattern_container_type& p);
    };

    field<TT1, Arch_A> field_1a;
    field<TT1, Arch_B> field_1b;
    field<TT2, Arch_A> field_2a;
    field<TT2, Arch_B> field_2b;
    field<TT3, Arch_A> field_3a;
    field<TT3, Arch_B> field_3b;

    parameters(ghex::context& c);

    void check_values();

  private:
    template<typename T, typename Arch>
    void fill_values(field<T, Arch>& f)
    {
        fill_values(f.raw_field, f.dom, Arch{});
    }

    template<typename T>
    void fill_values(
        ghex::test::util::memory<array_type<T, 3>>& m, domain_descriptor_type const& d, ghex::cpu);

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    template<typename T>
    void fill_values(
        ghex::test::util::memory<array_type<T, 3>>& m, domain_descriptor_type const& d, ghex::gpu)
    {
        fill_values(m, d, ghex::cpu{});
        m.clone_to_device();
    }
#endif

    template<typename T, typename Arch>
    bool check_values(field<T, Arch>& f)
    {
        return check_values(f.raw_field, f.dom, f.halos, Arch{});
    }

    template<typename T>
    bool check_values(ghex::test::util::memory<array_type<T, 3>>& m,
        domain_descriptor_type const& d, std::array<int, 6> const& halos, ghex::cpu);

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    template<typename T>
    bool check_values(ghex::test::util::memory<array_type<T, 3>>& m,
        domain_descriptor_type const& d, std::array<int, 6> const& halos, ghex::gpu)
    {
        m.clone_to_host();
        return check_values(m, d, halos, ghex::cpu{});
    }
#endif
};

// class for testing different exchange modes
// - serial (non-threaded)
//     - 1 communication object
//     - 2 communication objects (one per subdomain)
// - multi-threaded
//     - 1 thread per subdomain
//     - different thread launch mechanisms (thread, async, deferred)
template<typename T1, typename T2, typename T3, typename Arch1, typename Arch2>
struct test_exchange
{
    using params_type = parameters<T1, T2, T3, Arch1, Arch2>;
    using pattern_container_type = typename params_type::pattern_container_type;

    static void run(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        co = ghex::make_communication_object<pattern_container_type>(ctxt);
        co.exchange(params.field_1a.bi, params.field_1b.bi, params.field_2a.bi, params.field_2b.bi,
              params.field_3a.bi, params.field_3b.bi)
            .wait();
        params.check_values();
    }

    static void run_split(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        co1 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto        co2 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto        h1 = co1.exchange(params.field_1a.bi, params.field_2a.bi, params.field_3a.bi);
        auto        h2 = co2.exchange(params.field_1b.bi, params.field_2b.bi, params.field_3b.bi);
        while (!h1.is_ready() || !h2.is_ready())
        {
            h1.progress();
            h2.progress();
        }
        params.check_values();
    }

    static void run_mt(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        func = [&ctxt](auto... bis) {
            auto co = ghex::make_communication_object<pattern_container_type>(ctxt);
            co.exchange(bis...).wait();
        };
        std::vector<std::thread> threads;
        threads.push_back(
            std::thread{func, params.field_1a.bi, params.field_2a.bi, params.field_3a.bi});
        threads.push_back(
            std::thread{func, params.field_1b.bi, params.field_2b.bi, params.field_3b.bi});
        for (auto& t : threads) t.join();
        params.check_values();
    }

    static void run_mt_async(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        func = [&ctxt](auto... bis) {
            auto co = ghex::make_communication_object<pattern_container_type>(ctxt);
            co.exchange(bis...).wait();
        };
        auto policy = std::launch::async;
        // note: deferred launch policy does not work since it will deadlock in the func
        auto future1 =
            std::async(policy, func, params.field_1a.bi, params.field_2a.bi, params.field_3a.bi);
        auto future2 =
            std::async(policy, func, params.field_1b.bi, params.field_2b.bi, params.field_3b.bi);
        future1.wait();
        future2.wait();
        params.check_values();
    }

    static void run_mt_async_ret(ghex::context& ctxt) { run_mt_r(ctxt, std::launch::async); }

    static void run_mt_deferred_ret(ghex::context& ctxt) { run_mt_r(ctxt, std::launch::deferred); }

  private:
    template<typename Policy>
    static void run_mt_r(ghex::context& ctxt, Policy policy)
    {
        params_type params(ctxt);
        auto        func = [&ctxt](auto co, auto... bis) { return co->exchange(bis...); };
        auto        co1 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto        co2 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto        future1 = std::async(
            policy, func, &co1, params.field_1a.bi, params.field_2a.bi, params.field_3a.bi);
        auto future2 = std::async(
            policy, func, &co2, params.field_1b.bi, params.field_2b.bi, params.field_3b.bi);
        auto h1 = future1.get();
        auto h2 = future2.get();
        while (!h1.is_ready() || !h2.is_ready())
        {
            h1.progress();
            h2.progress();
        }
        params.check_values();
    }
};

// class for testing different exchange modes
// same as above but for fields of uniform data type T
// fields can be exchanged using runtime interface (i.e. begin/end iterators)
template<typename T, typename Arch1, typename Arch2>
struct test_exchange_vector
{
    using params_type = parameters<T, T, T, Arch1, Arch2>;
    using pattern_container_type = typename params_type::pattern_container_type;

    static void run(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        co = ghex::make_communication_object<pattern_container_type>(ctxt);
        std::vector<decltype(params.field_1a.bi)> fields1{
            params.field_1a.bi, params.field_2a.bi, params.field_3a.bi};
        std::vector<decltype(params.field_1b.bi)> fields2{
            params.field_1b.bi, params.field_2b.bi, params.field_3b.bi};
        co.exchange(fields1.begin(), fields1.end(), fields2.begin(), fields2.end()).wait();
        params.check_values();
    }

    static void run_split(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        co1 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto        co2 = ghex::make_communication_object<pattern_container_type>(ctxt);
        std::vector<decltype(params.field_1a.bi)> fields1{
            params.field_1a.bi, params.field_2a.bi, params.field_3a.bi};
        std::vector<decltype(params.field_1b.bi)> fields2{
            params.field_1b.bi, params.field_2b.bi, params.field_3b.bi};
        auto h1 = co1.exchange(fields1.begin(), fields1.end());
        auto h2 = co2.exchange(fields2.begin(), fields2.end());
        while (!h1.is_ready() || !h2.is_ready())
        {
            h1.progress();
            h2.progress();
        }
        params.check_values();
    }

    static void run_mt(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        func = [&ctxt](auto vec) {
            auto co = ghex::make_communication_object<pattern_container_type>(ctxt);
            co.exchange(vec.begin(), vec.end()).wait();
        };
        std::vector<decltype(params.field_1a.bi)> fields1{
            params.field_1a.bi, params.field_2a.bi, params.field_3a.bi};
        std::vector<decltype(params.field_1b.bi)> fields2{
            params.field_1b.bi, params.field_2b.bi, params.field_3b.bi};
        std::vector<std::thread> threads;
        threads.push_back(std::thread{func, fields1});
        threads.push_back(std::thread{func, fields2});
        for (auto& t : threads) t.join();
        params.check_values();
    }

    static void run_mt_async(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        func = [&ctxt](auto vec) {
            auto co = ghex::make_communication_object<pattern_container_type>(ctxt);
            co.exchange(vec.begin(), vec.end()).wait();
        };
        std::vector<decltype(params.field_1a.bi)> fields1{
            params.field_1a.bi, params.field_2a.bi, params.field_3a.bi};
        std::vector<decltype(params.field_1b.bi)> fields2{
            params.field_1b.bi, params.field_2b.bi, params.field_3b.bi};
        auto policy = std::launch::async;
        // note: deferred launch policy does not work since it will deadlock in the func
        auto future1 = std::async(policy, func, fields1);
        auto future2 = std::async(policy, func, fields2);
        future1.wait();
        future2.wait();
        params.check_values();
    }

    static void run_mt_async_ret(ghex::context& ctxt) { run_mt_r(ctxt, std::launch::async); }

    static void run_mt_deferred_ret(ghex::context& ctxt) { run_mt_r(ctxt, std::launch::deferred); }

  private:
    template<typename Policy>
    static void run_mt_r(ghex::context& ctxt, Policy policy)
    {
        params_type params(ctxt);
        auto func = [&ctxt](auto co, auto vec) { return co->exchange(vec.begin(), vec.end()); };
        auto co1 = ghex::make_communication_object<pattern_container_type>(ctxt);
        auto co2 = ghex::make_communication_object<pattern_container_type>(ctxt);
        std::vector<decltype(params.field_1a.bi)> fields1{
            params.field_1a.bi, params.field_2a.bi, params.field_3a.bi};
        std::vector<decltype(params.field_1b.bi)> fields2{
            params.field_1b.bi, params.field_2b.bi, params.field_3b.bi};
        auto future1 = std::async(policy, func, &co1, fields1);
        auto future2 = std::async(policy, func, &co2, fields2);
        auto h1 = future1.get();
        auto h2 = future2.get();
        while (!h1.is_ready() || !h2.is_ready())
        {
            h1.progress();
            h2.progress();
        }
        params.check_values();
    }
};

// specializaton if T1==T2==T3 but Arch1 != Arch2
template<typename T, typename Arch1, typename Arch2>
struct test_exchange<T, T, T, Arch1, Arch2> : public test_exchange_vector<T, Arch1, Arch2>
{
};

// specializaton if T1==T2==T3 and Arch1 == Arch2
template<typename T, typename Arch>
struct test_exchange<T, T, T, Arch, Arch> : public test_exchange_vector<T, Arch, Arch>
{
    using params_type = parameters<T, T, T, Arch, Arch>;
    using pattern_container_type = typename params_type::pattern_container_type;

    static void run(ghex::context& ctxt)
    {
        params_type params(ctxt);
        auto        co = ghex::make_communication_object<pattern_container_type>(ctxt);
        std::vector<decltype(params.field_1a.bi)> fields{params.field_1a.bi, params.field_1b.bi,
            params.field_2a.bi, params.field_2b.bi, params.field_3a.bi, params.field_3b.bi};
        co.exchange(fields.begin(), fields.end()).wait();
        params.check_values();
    }
};

//////////////////////////////
// actual tests follow here //
//////////////////////////////

TEST_F(mpi_test_fixture, exchange_host_host)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run_mt(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run_mt_async(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run_mt_async_ret(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::cpu>::run_mt_deferred_ret(ctxt);
    }
}

TEST_F(mpi_test_fixture, exchange_host_host_vector)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run_mt(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run_mt_async(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run_mt_async_ret(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::cpu>::run_mt_deferred_ret(ctxt);
    }
}

#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
TEST_F(mpi_test_fixture, exchange_device_device)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run(ctxt);
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run_mt(ctxt);
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run_mt_async(ctxt);
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run_mt_async_ret(ctxt);
        test_exchange<double, float, int, ghex::gpu, ghex::gpu>::run_mt_deferred_ret(ctxt);
    }
}

TEST_F(mpi_test_fixture, exchange_device_device_vector)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run(ctxt);
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run_mt(ctxt);
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run_mt_async(ctxt);
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run_mt_async_ret(ctxt);
        test_exchange<double, double, double, ghex::gpu, ghex::gpu>::run_mt_deferred_ret(ctxt);
    }
}

TEST_F(mpi_test_fixture, exchange_host_device)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run_mt(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run_mt_async(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run_mt_async_ret(ctxt);
        test_exchange<double, float, int, ghex::cpu, ghex::gpu>::run_mt_deferred_ret(ctxt);
    }
}

TEST_F(mpi_test_fixture, exchange_host_device_vector)
{
    using namespace ghex;
    EXPECT_TRUE((world_size == 1) || (world_size % 2 == 0));
    context ctxt(world, thread_safe);

    if (!thread_safe)
    {
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run_split(ctxt);
    }
    else
    {
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run_mt(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run_mt_async(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run_mt_async_ret(ctxt);
        test_exchange<double, double, double, ghex::cpu, ghex::gpu>::run_mt_deferred_ret(ctxt);
    }
}
#endif

// implementation

/* clang-format off */
template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
parameters<T1, T2, T3, Arch_A, Arch_B>::parameters(ghex::context& c)
: ctxt{c}
, g_last{local_ext[0] * 4 - 1, ((ctxt.size() - 1) / 2 + 1) * local_ext[1] - 1, local_ext[2] - 1}
, local_ext_buffer{local_ext[0] + 2 * offset[0], local_ext[1] + 2 * offset[1],
      local_ext[2] + 2 * offset[2]}
, max_memory{local_ext_buffer[0] * local_ext_buffer[1] * local_ext_buffer[2]}
, local_domains{
    domain_descriptor_type{
        ctxt.rank() * 2,
        std::array<int, 3>{
            ((ctxt.rank() % 2) * 2) * local_ext[0],
            (ctxt.rank() / 2) * local_ext[1],
            0},
        std::array<int, 3>{
            ((ctxt.rank() % 2) * 2 + 1) * local_ext[0] - 1,
            (ctxt.rank() / 2 + 1) * local_ext[1] - 1,
            local_ext[2] - 1}},
      domain_descriptor_type{
          ctxt.rank() * 2 + 1,
          std::array<int, 3>{
              ((ctxt.rank() % 2) * 2 + 1) * local_ext[0],
              (ctxt.rank() / 2) * local_ext[1],
              0},
          std::array<int, 3>{
              ((ctxt.rank() % 2) * 2 + 2) * local_ext[0] - 1,
              (ctxt.rank() / 2 + 1) * local_ext[1] - 1,
              local_ext[2] - 1}}}
, pattern1{std::make_unique<pattern_container_type>(ghex::make_pattern<ghex::structured::grid>(
      ctxt, halo_generator_type(g_first, g_last, halos_1, periodic), local_domains))}
, pattern2{std::make_unique<pattern_container_type>(ghex::make_pattern<ghex::structured::grid>(
      ctxt, halo_generator_type(g_first, g_last, halos_2, periodic), local_domains))}
, field_1a{max_memory, offset, local_ext_buffer, local_domains[0], halos_1, *pattern1}
, field_1b{max_memory, offset, local_ext_buffer, local_domains[1], halos_1, *pattern1}
, field_2a{max_memory, offset, local_ext_buffer, local_domains[0], halos_2, *pattern2}
, field_2b{max_memory, offset, local_ext_buffer, local_domains[1], halos_2, *pattern2}
, field_3a{max_memory, offset, local_ext_buffer, local_domains[0], halos_1, *pattern1}
, field_3b{max_memory, offset, local_ext_buffer, local_domains[1], halos_1, *pattern1}
{
    fill_values(field_1a);
    fill_values(field_1b);
    fill_values(field_2a);
    fill_values(field_2b);
    fill_values(field_3a);
    fill_values(field_3b);
}
/* clang-format on */

template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
template<typename T>
void
parameters<T1, T2, T3, Arch_A, Arch_B>::fill_values(
    ghex::test::util::memory<array_type<T, 3>>& m, domain_descriptor_type const& d, ghex::cpu)
{
    for (int z = 0; z < local_ext[2]; ++z)
        for (int y = 0; y < local_ext[1]; ++y)
            for (int x = 0; x < local_ext[0]; ++x)
                m[(x + offset[0]) + local_ext_buffer[0] *
                                        ((y + offset[1]) + local_ext_buffer[1] * (z + offset[2]))] =
                    array_type<T, 3>{
                        (T)(x + d.first()[0]), (T)(y + d.first()[1]), (T)(z + d.first()[2])};
}

template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
void
parameters<T1, T2, T3, Arch_A, Arch_B>::check_values()
{
    EXPECT_TRUE(check_values(field_1a));
    EXPECT_TRUE(check_values(field_1b));
    EXPECT_TRUE(check_values(field_2a));
    EXPECT_TRUE(check_values(field_2b));
    EXPECT_TRUE(check_values(field_3a));
    EXPECT_TRUE(check_values(field_3b));
}

template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
template<typename T>
bool
parameters<T1, T2, T3, Arch_A, Arch_B>::check_values(ghex::test::util::memory<array_type<T, 3>>& m,
    domain_descriptor_type const& d, std::array<int, 6> const& halos, ghex::cpu)
{
    bool       passed = true;
    const auto i = d.domain_id() % 2;
    const auto rank = ctxt.rank();
    const auto size = ctxt.size();

    int xl = -halos[0];
    int hxl = halos[0];
    int hxr = halos[1];
    // hack begin: make it work with 1 rank (works with even number of ranks otherwise)
    if (i == 0 && size == 1)
    {
        xl = 0;
        hxl = 0;
    }
    if (i == 1 && size == 1) { hxr = 0; }
    // hack end
    for (int x = d.first()[0] - hxl; x <= d.last()[0] + hxr; ++x, ++xl)
    {
        if (i == 0 && x < d.first()[0] && !periodic[0]) continue;
        if (i == 1 && x > d.last()[0] && !periodic[0]) continue;
        T x_wrapped =
            (((x - g_first[0]) + (g_last[0] - g_first[0] + 1)) % (g_last[0] - g_first[0] + 1) +
                g_first[0]);
        int yl = -halos[2];
        for (int y = d.first()[1] - halos[2]; y <= d.last()[1] + halos[3]; ++y, ++yl)
        {
            if (d.domain_id() < 2 && y < d.first()[1] && !periodic[1]) continue;
            if (d.domain_id() > size - 3 && y > d.last()[1] && !periodic[1]) continue;
            T y_wrapped =
                (((y - g_first[1]) + (g_last[1] - g_first[1] + 1)) % (g_last[1] - g_first[1] + 1) +
                    g_first[1]);
            int zl = -halos[4];
            for (int z = d.first()[2] - halos[4]; z <= d.last()[2] + halos[5]; ++z, ++zl)
            {
                if (z < d.first()[2] && !periodic[2]) continue;
                if (z > d.last()[2] && !periodic[2]) continue;
                T z_wrapped = (((z - g_first[2]) + (g_last[2] - g_first[2] + 1)) %
                                   (g_last[2] - g_first[2] + 1) +
                               g_first[2]);

                const auto& value =
                    m[(xl + offset[0]) +
                        local_ext_buffer[0] *
                            ((yl + offset[1]) + local_ext_buffer[1] * (zl + offset[2]))];
                if (value[0] != x_wrapped || value[1] != y_wrapped || value[2] != z_wrapped)
                {
                    passed = false;
                    std::cout << "(" << xl << ", " << yl << ", " << zl
                              << ") values found != expected: "
                              << "(" << value[0] << ", " << value[1] << ", " << value[2] << ") != "
                              << "(" << x_wrapped << ", " << y_wrapped << ", " << z_wrapped << ") "
                              << i << "  " << rank << std::endl;
                }
            }
        }
    }
    return passed;
}

template<typename T1, typename T2, typename T3, typename Arch_A, typename Arch_B>
template<typename T, typename Arch>
parameters<T1, T2, T3, Arch_A, Arch_B>::field<T, Arch>::field(

    int size, std::array<int, 3> const& off, std::array<int, 3> const& ext,
    domain_descriptor_type& d, std::array<int, 6> const halos_, pattern_container_type& p)
//: raw_field{(unsigned int)size}
// no freeing of (pinned) GPU memory because of MPICH bug on daint...
: raw_field{(unsigned int)size, T{}, true}
, ghex_field{wrap(raw_field, d, off, ext, Arch{})}
, dom{d}
, halos{halos_}
, pattern{p}
, bi{pattern(ghex_field)}
{
}
