/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#include <sstream>
#include <utility>

#include <mpi.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <ghex/arch_traits.hpp>
#include <ghex/communication_object_2.hpp>
#include <ghex/transport_layer/context.hpp>

#include <ghex/transport_layer/mpi/context.hpp>

#include <ghex/structured/pattern.hpp>
#include <ghex/structured/regular/domain_descriptor.hpp>
#include <ghex/structured/regular/halo_generator.hpp>
#include <ghex/structured/regular/field_descriptor.hpp>

#include <ghex/bindings/python/type_list.hpp>
#include <ghex/bindings/python/binding_registry.hpp>
#include <ghex/bindings/python/utils/register_wrapper.hpp>
#include <ghex/bindings/python/utils/mpi_comm_shim.hpp>

namespace ghex4py = gridtools::ghex::bindings::python;

namespace py = pybind11;

#include <sstream>


template <typename GridType, typename Transport, typename HaloGenerator, typename DomainRange>
void register_make_pattern(pybind11::module& m) {
    using context_type = typename gridtools::ghex::tl::context_factory<Transport>::context_type&;

    auto make_pattern_wrapper = [] (context_type& context, HaloGenerator& hgen, std::vector<DomainRange>& d_range) {
        return gridtools::ghex::make_pattern<GridType>(context, hgen, d_range);
    };
    using pattern_type = typename std::result_of<decltype(make_pattern_wrapper)(context_type&, HaloGenerator&, std::vector<DomainRange>&)>::type;

    m.def("make_pattern", make_pattern_wrapper);

    // todo: move + cleanup
    using arch = gridtools::ghex::cpu;
    using domain_id_type = int;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<domain_id_type, std::integral_constant<int, 3>>;
    using array_type = std::array<int, 3>;
    auto wrapper = [] (const domain_descriptor_type& dom, double* data, const array_type& offsets, const array_type& extents) {
        return gridtools::ghex::wrap_field<arch, ::gridtools::layout_map<0,1,2>>(dom, data, offsets, extents);
    };
    using field_t = decltype(wrapper(std::declval<const domain_descriptor_type&>(), std::declval<double*>(), std::declval<const array_type&>(), std::declval<const array_type&>()));

    py::class_<pattern_type>(m, "PatternContainer")
        .def("__call__", &pattern_type::template operator()<field_t>);

    using buffer_info_type = decltype(std::declval<pattern_type>()(std::declval<field_t&>()));
    py::class_<buffer_info_type>(m, "BufferInfo");
}


template <typename GridType, typename Transport, typename HaloGenerator, typename Domain>
void register_make_communication_object(pybind11::module& m) {
    using grid_type = typename GridType::template type<Domain>;
    using domain_id_type = typename Domain::domain_id_type;


    using context_type = typename gridtools::ghex::tl::context_factory<Transport>::context_type;
    using pattern_type = decltype(gridtools::ghex::make_pattern<GridType>(std::declval<context_type&>(), std::declval<HaloGenerator&>(), std::declval<std::vector<Domain>&>()));
    using communicator_type = typename context_type::communicator_type;
    using communication_object_type = gridtools::ghex::communication_object<communicator_type, grid_type, domain_id_type>;


    // todo: just for getting something runnable
    using arch = gridtools::ghex::cpu;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<domain_id_type, std::integral_constant<int, 3>>;
    using array_type = std::array<int, 3>;
    auto wrapper = [] (const domain_descriptor_type& dom, double* data, const array_type& offsets, const array_type& extents) {
        return gridtools::ghex::wrap_field<arch, ::gridtools::layout_map<0,1,2>>(dom, data, offsets, extents);
    };
    using field_t = decltype(wrapper(std::declval<const domain_descriptor_type&>(), std::declval<double*>(), std::declval<const array_type&>(), std::declval<const array_type&>()));

    using buffer_info_type = decltype(std::declval<pattern_type>()(std::declval<field_t&>()));

    py::class_<communication_object_type>(m, "CommunicationObject")
        .def(py::init([] (communicator_type c) {
            return gridtools::ghex::make_communication_object<pattern_type>(c); }))
        .def("exchange", [] (communication_object_type& co, buffer_info_type& b) { return co.exchange(b); });

    py::class_<typename communication_object_type::handle_type>(m, "CommunicationHandle")
        .def("wait", &communication_object_type::handle_type::wait);

}

using gridtools::ghex::bindings::python::utils::register_wrapper;

PYBIND11_MODULE(ghex_py_bindings, m) {
    gridtools::ghex::bindings::python::BindingRegistry::get_instance().set_initialized(m);

    m.doc() = "pybind11 ghex bindings"; // optional module docstring

    m.def_submodule("utils").def("hash_str", [] (const std::string& val) { return std::hash<std::string>()(val); });

    using domain_id_type = int;
    using domain_descriptor_type = gridtools::ghex::structured::regular::domain_descriptor<domain_id_type, std::integral_constant<int, 3>>;

    pybind11::class_<mpi_comm_shim> mpi_comm(m, "mpi_comm");
    mpi_comm.def(pybind11::init<>());

    py::class_<gridtools::ghex::coordinate<std::array<int, 3>>>(m, "Coordinate3d");

    /*
     * Pattern
     */
    register_make_pattern<gridtools::ghex::structured::grid, ghex4py::type_list::transport, ghex4py::type_list::halo_generator_type, domain_descriptor_type>(m);

    /*
     * Communication object
     */
    register_make_communication_object<gridtools::ghex::structured::grid, ghex4py::type_list::transport, ghex4py::type_list::halo_generator_type, domain_descriptor_type>(m);
}