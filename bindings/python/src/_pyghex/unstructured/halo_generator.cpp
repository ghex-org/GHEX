/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <gridtools/common/for_each.hpp>

#include <register_class.hpp>
#include <unstructured/halo_generator.hpp>

namespace pyghex
{
namespace unstructured
{
void
register_halo_generator(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, halo_generator_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using global_index_type = typename type::global_index_type;
            using halo = typename type::halo;

            auto _halo_generator = register_class<type>(m);
            /*auto _halo = */register_class<halo>(m);

            _halo_generator
                .def(pybind11::init<>(), "Create a halo generator")
                .def(pybind11::init([](const std::vector<global_index_type>& gids){ return type{gids};}))
                .def("__call__", &type::operator());
        });
}

} // namespace unstructured
} // namespace pyghex
