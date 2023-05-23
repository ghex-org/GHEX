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
#include <ghex/structured/regular/halo_generator.hpp>
#ifdef GHEX_CUDACC
#include <ghex/device/cuda/runtime.hpp>
#endif

#include "../../util/memory.hpp"
#include <gridtools/common/array.hpp>
#include <thread>
#include <array>

struct simulation_1
{
    template<typename T, std::size_t N>
    using array_type = gridtools::array<T, N>;
    using T1 = double;
    using T2 = float;
    using T3 = int;
    using TT1 = array_type<T1, 3>;
    using TT2 = array_type<T2, 3>;
    using TT3 = array_type<T3, 3>;

    using domain_descriptor_type =
        ghex::structured::regular::domain_descriptor<int, std::integral_constant<int, 3>>;
    using halo_generator_type =
        ghex::structured::regular::halo_generator<int, std::integral_constant<int, 3>>;
    template<typename T, typename Arch, int... Is>
    using field_descriptor_type = ghex::structured::regular::field_descriptor<T, Arch,
        domain_descriptor_type, gridtools::layout_map<Is...>>;

    // decomposition: 4 domains in x-direction, 1 domain in z-direction, rest in y-direction
    //                each MPI rank owns two domains: either first or last two domains in x-direction
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

    ghex::context                       ctxt;
    const std::array<int, 3>            local_ext;
    const std::array<bool, 3>           periodic;
    const std::array<int, 3>            g_first;
    const std::array<int, 3>            g_last;
    const std::array<int, 3>            offset;
    const std::array<int, 3>            local_ext_buffer;
    const int                           max_memory;
    ghex::test::util::memory<TT1>       field_1a_raw;
    ghex::test::util::memory<TT1>       field_1b_raw;
    ghex::test::util::memory<TT2>       field_2a_raw;
    ghex::test::util::memory<TT2>       field_2b_raw;
    ghex::test::util::memory<TT3>       field_3a_raw;
    ghex::test::util::memory<TT3>       field_3b_raw;
    std::vector<domain_descriptor_type> local_domains;
    std::array<int, 6>                  halos;
    halo_generator_type                 halo_gen;
    using pattern_type = std::remove_reference_t<decltype(
        ghex::make_pattern<ghex::structured::grid>(ctxt, halo_gen, local_domains))>;
    pattern_type                                   pattern;
    field_descriptor_type<TT1, ghex::cpu, 2, 1, 0> field_1a;
    field_descriptor_type<TT1, ghex::cpu, 2, 1, 0> field_1b;
    field_descriptor_type<TT2, ghex::cpu, 2, 1, 0> field_2a;
    field_descriptor_type<TT2, ghex::cpu, 2, 1, 0> field_2b;
    field_descriptor_type<TT3, ghex::cpu, 2, 1, 0> field_3a;
    field_descriptor_type<TT3, ghex::cpu, 2, 1, 0> field_3b;
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    field_descriptor_type<TT1, ghex::gpu, 2, 1, 0> field_1a_gpu;
    field_descriptor_type<TT1, ghex::gpu, 2, 1, 0> field_1b_gpu;
    field_descriptor_type<TT2, ghex::gpu, 2, 1, 0> field_2a_gpu;
    field_descriptor_type<TT2, ghex::gpu, 2, 1, 0> field_2b_gpu;
    field_descriptor_type<TT3, ghex::gpu, 2, 1, 0> field_3a_gpu;
    field_descriptor_type<TT3, ghex::gpu, 2, 1, 0> field_3b_gpu;
#endif
    bool                                                 mt;
    std::vector<ghex::generic_bulk_communication_object> cos;

    simulation_1(bool multithread = false)
    : ctxt{MPI_COMM_WORLD, multithread}
    , local_ext{4, 3, 2}
    , periodic{true, true, true}
    , g_first{0, 0, 0}
    , g_last{local_ext[0] * 4 - 1, ((ctxt.size() - 1) / 2 + 1) * local_ext[1] - 1, local_ext[2] - 1}
    , offset{3, 3, 3}
    , local_ext_buffer{local_ext[0] + 2 * offset[0], local_ext[1] + 2 * offset[1],
          local_ext[2] + 2 * offset[2]}
    , max_memory{local_ext_buffer[0] * local_ext_buffer[1] * local_ext_buffer[2]}
    , field_1a_raw(max_memory)
    , field_1b_raw(max_memory)
    , field_2a_raw(max_memory)
    , field_2b_raw(max_memory)
    , field_3a_raw(max_memory)
    , field_3b_raw(max_memory)
    , local_domains{domain_descriptor_type{ctxt.rank() * 2,
                        std::array<int, 3>{((ctxt.rank() % 2) * 2) * local_ext[0],
                            (ctxt.rank() / 2) * local_ext[1], 0},
                        std::array<int, 3>{((ctxt.rank() % 2) * 2 + 1) * local_ext[0] - 1,
                            (ctxt.rank() / 2 + 1) * local_ext[1] - 1, local_ext[2] - 1}},
          domain_descriptor_type{ctxt.rank() * 2 + 1,
              std::array<int, 3>{
                  ((ctxt.rank() % 2) * 2 + 1) * local_ext[0], (ctxt.rank() / 2) * local_ext[1], 0},
              std::array<int, 3>{((ctxt.rank() % 2) * 2 + 2) * local_ext[0] - 1,
                  (ctxt.rank() / 2 + 1) * local_ext[1] - 1, local_ext[2] - 1}}}
    , halos{2, 2, 2, 2, 2, 2}
    , halo_gen(g_first, g_last, halos, periodic)
    , pattern{ghex::make_pattern<ghex::structured::grid>(ctxt, halo_gen, local_domains)}
    , field_1a{ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[0], field_1a_raw.data(), offset, local_ext_buffer)}
    , field_1b{ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[1], field_1b_raw.data(), offset, local_ext_buffer)}
    , field_2a{ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[0], field_2a_raw.data(), offset, local_ext_buffer)}
    , field_2b{ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[1], field_2b_raw.data(), offset, local_ext_buffer)}
    , field_3a{ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[0], field_3a_raw.data(), offset, local_ext_buffer)}
    , field_3b
    {
        ghex::wrap_field<ghex::cpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[1], field_3b_raw.data(), offset, local_ext_buffer)
    }
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
    , field_1a_gpu{ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
          local_domains[0], field_1a_raw.device_data(), offset, local_ext_buffer)},
        field_1b_gpu{ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[1], field_1b_raw.device_data(), offset, local_ext_buffer)},
        field_2a_gpu{ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[0], field_2a_raw.device_data(), offset, local_ext_buffer)},
        field_2b_gpu{ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[1], field_2b_raw.device_data(), offset, local_ext_buffer)},
        field_3a_gpu{ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[0], field_3a_raw.device_data(), offset, local_ext_buffer)},
        field_3b_gpu
    {
        ghex::wrap_field<ghex::gpu, gridtools::layout_map<2, 1, 0>>(
            local_domains[1], field_3b_raw.device_data(), offset, local_ext_buffer)
    }
#endif
    , mt{multithread}
    {
        fill_values(local_domains[0], field_1a);
        fill_values(local_domains[1], field_1b);
        fill_values(local_domains[0], field_2a);
        fill_values(local_domains[1], field_2b);
        fill_values(local_domains[0], field_3a);
        fill_values(local_domains[1], field_3b);
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
        field_1a_raw.clone_to_device();
        field_1b_raw.clone_to_device();
        field_2a_raw.clone_to_device();
        field_2b_raw.clone_to_device();
        field_3a_raw.clone_to_device();
        field_3b_raw.clone_to_device();
#endif

        if (!mt)
        {
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
            auto bco = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::gpu, 2, 1, 0>>(ctxt);

            bco.add_field(pattern(field_1a_gpu));
            bco.add_field(pattern(field_1b_gpu));
            bco.add_field(pattern(field_2a_gpu));
            bco.add_field(pattern(field_2b_gpu));
            bco.add_field(pattern(field_3a_gpu));
            bco.add_field(pattern(field_3b_gpu));
#else
            auto bco = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::cpu, 2, 1, 0>>(ctxt);

            bco.add_field(pattern(field_1a));
            bco.add_field(pattern(field_1b));
            bco.add_field(pattern(field_2a));
            bco.add_field(pattern(field_2b));
            bco.add_field(pattern(field_3a));
            bco.add_field(pattern(field_3b));
#endif
            cos.emplace_back(std::move(bco));
        }
        else
        {
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
            auto bco0 = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::gpu, 2, 1, 0>>(ctxt);
            bco0.add_field(pattern(field_1a_gpu));
            bco0.add_field(pattern(field_2a_gpu));
            bco0.add_field(pattern(field_3a_gpu));

            auto bco1 = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::gpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::gpu, 2, 1, 0>>(ctxt);
            bco1.add_field(pattern(field_1b_gpu));
            bco1.add_field(pattern(field_2b_gpu));
            bco1.add_field(pattern(field_3b_gpu));
#else
            auto bco0 = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::cpu, 2, 1, 0>>(ctxt);
            bco0.add_field(pattern(field_1a));
            bco0.add_field(pattern(field_2a));
            bco0.add_field(pattern(field_3a));

            auto bco1 = ghex::bulk_communication_object<ghex::structured::rma_range_generator,
                pattern_type, field_descriptor_type<TT1, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT2, ghex::cpu, 2, 1, 0>,
                field_descriptor_type<TT3, ghex::cpu, 2, 1, 0>>(ctxt);
            bco1.add_field(pattern(field_1b));
            bco1.add_field(pattern(field_2b));
            bco1.add_field(pattern(field_3b));
#endif
            cos.emplace_back(std::move(bco0));
            cos.emplace_back(std::move(bco1));
        }
    }

    void exchange()
    {
        if (!mt) { cos[0].exchange().wait(); }
        else
        {
            std::vector<std::thread> threads;
            threads.push_back(std::thread{[this]() -> void { cos[0].exchange().wait(); }});
            threads.push_back(std::thread{[this]() -> void { cos[1].exchange().wait(); }});
            for (auto& t : threads) t.join();
        }
    }

    bool check()
    {
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
        field_1a_raw.clone_to_host();
        field_1b_raw.clone_to_host();
        field_2a_raw.clone_to_host();
        field_2b_raw.clone_to_host();
        field_3a_raw.clone_to_host();
        field_3b_raw.clone_to_host();
#endif
        bool passed = true;
        passed = passed && test_values(local_domains[0], field_1a);
        passed = passed && test_values(local_domains[1], field_1b);
        passed = passed && test_values(local_domains[0], field_2a);
        passed = passed && test_values(local_domains[1], field_2b);
        passed = passed && test_values(local_domains[0], field_3a);
        passed = passed && test_values(local_domains[1], field_3b);
        return passed;
    }

  private:
    template<typename Field>
    void fill_values(const domain_descriptor_type& d, Field& f)
    {
        using T = typename Field::value_type::value_type;
        int xl = 0;
        for (int x = d.first()[0]; x <= d.last()[0]; ++x, ++xl)
        {
            int yl = 0;
            for (int y = d.first()[1]; y <= d.last()[1]; ++y, ++yl)
            {
                int zl = 0;
                for (int z = d.first()[2]; z <= d.last()[2]; ++z, ++zl)
                { f(xl, yl, zl) = array_type<T, 3>{(T)x, (T)y, (T)z}; }
            }
        }
    }

    template<typename Field>
    bool test_values(const domain_descriptor_type& d, const Field& f)
    {
        using T = typename Field::value_type::value_type;
        bool      passed = true;
        const int i = d.domain_id() % 2;
        int       rank = ctxt.rank();
        int       size = ctxt.size();

        int xl = -halos[0];
        int hxl = halos[0];
        int hxr = halos[1];
        // hack begin: make it work with 1 rank (works with even number of ranks otherwise)
        if (i == 0 && size == 1) //comm.rank()%2 == 0 && comm.rank()+1 == comm.size())
        {
            xl = 0;
            hxl = 0;
        }
        if (i == 1 && size == 1) //comm.rank()%2 == 0 && comm.rank()+1 == comm.size())
        { hxr = 0; }
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
                T   y_wrapped = (((y - g_first[1]) + (g_last[1] - g_first[1] + 1)) %
                                   (g_last[1] - g_first[1] + 1) +
                               g_first[1]);
                int zl = -halos[4];
                for (int z = d.first()[2] - halos[4]; z <= d.last()[2] + halos[5]; ++z, ++zl)
                {
                    if (z < d.first()[2] && !periodic[2]) continue;
                    if (z > d.last()[2] && !periodic[2]) continue;
                    T z_wrapped = (((z - g_first[2]) + (g_last[2] - g_first[2] + 1)) %
                                       (g_last[2] - g_first[2] + 1) +
                                   g_first[2]);

                    const auto& value = f(xl, yl, zl);
                    if (value[0] != x_wrapped || value[1] != y_wrapped || value[2] != z_wrapped)
                    {
                        passed = false;
                        std::cout << "(" << xl << ", " << yl << ", " << zl
                                  << ") values found != expected: "
                                  << "(" << value[0] << ", " << value[1] << ", " << value[2]
                                  << ") != "
                                  << "(" << x_wrapped << ", " << y_wrapped << ", " << z_wrapped
                                  << ") " //<< std::endl;
                                  << i << "  " << rank << std::endl;
                    }
                }
            }
        }
        return passed;
    }
};

TEST_F(mpi_test_fixture, rma_exchange)
{
    simulation_1 sim(thread_safe);
    sim.exchange();
    sim.exchange();
    sim.exchange();
    EXPECT_TRUE(sim.check());
}
