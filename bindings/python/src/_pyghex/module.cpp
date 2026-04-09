/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include <nanobind/nanobind.h>

namespace pyghex
{
void register_config(nanobind::module_& m);
void register_mpi(nanobind::module_& m);
void register_context(nanobind::module_& m);
void register_py_dtype_to_cpp_name(nanobind::module_& m);

namespace structured
{
namespace regular
{
void register_domain_descriptor(nanobind::module_& m);
void register_halo_generator(nanobind::module_& m);
void register_field_descriptor(nanobind::module_& m);
void register_pattern(nanobind::module_& m);
void register_communication_object(nanobind::module_& m);
} // namespace regular
} // namespace structured

namespace unstructured
{
void register_domain_descriptor(nanobind::module_& m);
void register_halo_generator(nanobind::module_& m);
void register_field_descriptor(nanobind::module_& m);
void register_pattern(nanobind::module_& m);
void register_communication_object(nanobind::module_& m);
} // namespace unstructured

} // namespace pyghex

NB_MODULE(pyghex, m)
{
    m.doc() = "nanobind ghex bindings";

    pyghex::register_config(m);
    pyghex::register_mpi(m);
    pyghex::register_context(m);
    pyghex::register_py_dtype_to_cpp_name(m);

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
}
