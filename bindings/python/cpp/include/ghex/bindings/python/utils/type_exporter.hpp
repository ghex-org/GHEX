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
#pragma once

#include <pybind11/pybind11.h>

#include <gridtools/meta.hpp>
#include <gridtools/common/generic_metafunctions/for_each.hpp>
#include <ghex/bindings/python/binding_registry.hpp>
#include <ghex/bindings/python/utils/demangle.hpp>

template <typename T, class Dummy = void>
struct type_exporter {};

#define __GHEX_PYBIND11_EXPORT_TYPE(unique, type_exporter, specializations) \
static void* _register_##unique() { \
    gridtools::ghex::bindings::python::BindingRegistry::get_instance().register_exporter([] (pybind11::module_& m) { \
        gridtools::for_each<gridtools::meta::transform<type_exporter, specializations>>([&m] (auto registrator) { \
            using type = gridtools::meta::first<decltype(registrator)>; \
            auto type_name = gridtools::ghex::bindings::python::utils::demangle<type>(); \
            auto py_class = pybind11::class_<type>(m, type_name.c_str()) \
                .def_property_readonly_static("__cpp_type__", [type_name] (const pybind11::object&) { return type_name; }); \
            registrator(m, py_class); \
        }); \
    }); \
    return nullptr; \
} \
static void* _register_dummy_##unique = _register_##unique();

#define _GHEX_PYBIND11_EXPORT_TYPE(unique, type_exporter, specializations) __GHEX_PYBIND11_EXPORT_TYPE(unique, type_exporter, specializations)

#define GHEX_PYBIND11_EXPORT_TYPE(type_exporter, specializations) _GHEX_PYBIND11_EXPORT_TYPE(__COUNTER__, type_exporter, specializations)


#define __GHEX_PYBIND11_EXPORT(unique, exporter, args) \
static void* _register_##unique() { \
    gridtools::ghex::bindings::python::BindingRegistry::get_instance().register_exporter([] (pybind11::module_& m) { \
        gridtools::for_each<gridtools::meta::transform<exporter, args>>([&m] (auto registrator) { \
            registrator(m); \
        }); \
    }); \
    return nullptr; \
} \
static void* _register_dummy_##unique = _register_##unique();

#define _GHEX_PYBIND11_EXPORT(unique, exporter, args) __GHEX_PYBIND11_EXPORT(unique, exporter, args)

#define GHEX_PYBIND11_EXPORT(exporter, args) _GHEX_PYBIND11_EXPORT(__COUNTER__, exporter, args)