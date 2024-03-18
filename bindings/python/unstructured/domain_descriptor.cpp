/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <gridtools/common/for_each.hpp>

#include <util/demangle.hpp>
#include <unstructured/domain_descriptor.hpp>

namespace pyghex
{
namespace unstructured
{
void
register_domain_descriptor(pybind11::module& m)
{
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, domain_descriptor_specializations>>(
        [&m](auto l)
        {
            using namespace std::string_literals;
            using namespace pybind11::literals;

            using type = gridtools::meta::first<decltype(l)>;
            using domain_id_type = typename type::domain_id_type;
            using global_index_type = typename type::global_index_type;
            using local_index_type = typename type::local_index_type;

            auto type_name = util::demangle<type>();
            pybind11::class_<type>(m, type_name.c_str())
                .def(pybind11::init(
                    [](domain_id_type id, const std::vector<global_index_type>& gids,
                        const std::vector<local_index_type>& halo_lids) {
                        return type{id, gids.begin(), gids.end(), halo_lids.begin(),
                            halo_lids.end()};
                    }))
                .def("domain_id", &type::domain_id, "Returns the domain id")
                .def("size", &type::size, "Returns the size")
                .def("inner_size", &type::inner_size, "Returns the inner size")
                .def(
                    "indices",
                    [](const type& d) -> std::vector<global_index_type> { return d.gids(); },
                    "Returns the indices")
                .def_property_readonly_static("__cpp_type__",
                    [type_name](const pybind11::object&) { return type_name; });
        });
}

} // namespace unstructured
} // namespace pyghex
