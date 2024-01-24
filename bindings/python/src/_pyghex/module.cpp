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

    pyghex::register_config(m);
    pyghex::register_mpi(m);
    pyghex::register_context(m);

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
