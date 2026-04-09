/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <array>
#include <tuple>

#include <register_class.hpp>
#include <structured/regular/halo_generator.hpp>

#include <nanobind/nanobind.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/array.h>
#include <nanobind/stl/vector.h>

namespace pyghex
{
namespace structured
{
namespace regular
{

void
register_halo_generator(nanobind::module_& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, halo_generator_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace nanobind::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using dimension = typename type::dimension;
            using array = std::array<int, dimension::value>;
            using halo_array = std::array<int, 2 * dimension::value>;
            using periodic_array = std::array<bool, dimension::value>;
            using box = typename type::box;
            using box2 = typename type::box2;

            auto _halo_generator = register_class<type>(m);
            auto _box = register_class<box>(m);
            auto _box2 = register_class<box2>(m);

            _halo_generator
                .def(nanobind::init<array, array, halo_array, periodic_array>(), "first"_a,
                    "last"_a, "halos"_a, "periodic"_a, "Create a halo generator")
                .def("__call__",
                    [](const type& halo_gen, const typename type::domain_type& domain)
                    {
                        nanobind::list result;
                        for (const auto& halo : halo_gen(domain))
                            result.append(nanobind::cast(halo));
                        return result;
                    });

            _box2.def_prop_ro("local", [](const box2& b) { return b.local(); })
                .def_prop_ro("global_", [](const box2& b) { return b.global(); });

            _box.def_prop_ro("first",
                    [](const box& b)
                    {
                        auto first = b.first();
                        return static_cast<typename decltype(first)::array_type>(first);
                    })
                .def_prop_ro("last",
                    [](const box& b)
                    {
                        auto last = b.last();
                        return static_cast<typename decltype(last)::array_type>(last);
                    });
        });
}

} // namespace regular
} // namespace structured
} // namespace pyghex
