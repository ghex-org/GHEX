/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <pybind11/pybind11.h>
#include <gridtools/common/for_each.hpp>
#include <context_shim.hpp>
#include <unstructured/communication_object.hpp>
#include <unstructured/pattern.hpp>
#include <unstructured/domain_descriptor.hpp>

namespace py = pybind11;

namespace pyghex
{
void register_config(pybind11::module& m);
void register_mpi(pybind11::module& m);
void register_context(pybind11::module& m);

namespace structured
{
namespace regular
{
void register_domain_descriptor(pybind11::module& m);
void register_halo_generator(pybind11::module& m);
void register_field_descriptor(pybind11::module& m);
void register_pattern(pybind11::module& m);
void register_communication_object(pybind11::module& m);
} // namespace regular
} // namespace structured

namespace unstructured
{
void register_domain_descriptor(pybind11::module& m);
void register_halo_generator(pybind11::module& m);
void register_field_descriptor(pybind11::module& m);
void register_pattern(pybind11::module& m);
void register_communication_object(pybind11::module& m);
} // namespace unstructured

} // namespace pyghex

PYBIND11_MODULE(_pyghex, m)
{
    m.doc() = "pybind11 ghex bindings"; // optional module docstring

    //m.def_submodule("util")
    //    .def("hash_str", [] (const std::string& val) { return std::hash<std::string>()(val); })
    //    .def("mpi_library_version", [] () {
    //        int resultlen;
    //        char version[MPI_MAX_LIBRARY_VERSION_STRING];
    //        MPI_Get_library_version(version, &resultlen);
    //        return std::string(version);
    //    })
    //    ;

    pyghex::register_config(m);
    pyghex::register_mpi(m);
    pyghex::register_context(m);

    //auto structured = m.def_submodule("structured");
    //auto regular = structured.def_submodule("regular");

    pyghex::structured::regular::register_domain_descriptor(m);
    pyghex::structured::regular::register_halo_generator(m);
    pyghex::structured::regular::register_field_descriptor(m);
    pyghex::structured::regular::register_pattern(m);
    pyghex::structured::regular::register_communication_object(m);

    pyghex::unstructured::register_domain_descriptor(m);
    pyghex::unstructured::register_halo_generator(m);
    pyghex::unstructured::register_field_descriptor(m);
    pyghex::unstructured::register_pattern(m);
    pyghex::unstructured::register_communication_object(m);

    m.def("expose_cpp_ptr", [](pyghex::context_shim* obj) {return reinterpret_cast<uintptr_t>(&obj->m);} );
    
    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, pyghex::unstructured::communication_object_specializations>>(
        [&m](auto l)
        {
            using type = gridtools::meta::first<decltype(l)>;
            m.def("expose_cpp_ptr", [](type* obj) {return reinterpret_cast<uintptr_t>(obj);} );
        });

    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, pyghex::unstructured::make_pattern_traits_specializations>>(
        [&m](auto l)
        {
            using type = gridtools::meta::first<decltype(l)>;
            using halo_gen = typename type::halo_gen;
            using domain_range = typename type::domain_range;
            using grid_type = ghex::unstructured::grid;
            using pattern_container =
                decltype(ghex::make_pattern<grid_type>(std::declval<ghex::context&>(),
                    std::declval<halo_gen&>(), std::declval<domain_range&>()));
            m.def("expose_cpp_ptr", [](pattern_container* obj) {return reinterpret_cast<uintptr_t>(obj);} );
        });

    gridtools::for_each<
        gridtools::meta::transform<gridtools::meta::list, pyghex::unstructured::domain_descriptor_specializations>>(
        [&m](auto l)
        {
            using type = gridtools::meta::first<decltype(l)>;
            m.def("expose_cpp_ptr", [](type* obj) {return reinterpret_cast<uintptr_t>(obj);} );
        });
}
