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

#include <cassert>
#include <functional>

#include <pybind11/pybind11.h>

namespace gridtools {
namespace ghex {
namespace bindings {
namespace python {

class BindingRegistry
{
public: // member type
    using exporter_type = std::function<void(pybind11::module_&)>;

public: // ctors
    BindingRegistry(BindingRegistry const&) = delete;
    void operator=(BindingRegistry const&) = delete;

private: // private ctors
    BindingRegistry() {}

    ~BindingRegistry() {
        assert(m_initialized == true);
    }

public: // member functions
    static BindingRegistry& get_instance()
    {
        static BindingRegistry instance;
        return instance;
    }

    void set_initialized(pybind11::module_& m) {
        assert(m_initialized == false);
        m_initialized = true;

        for (auto& exporter : m_exporter) {
            exporter(m);
        }
    }

    void register_exporter(const exporter_type& exporter) {
        if (m_initialized) {
            throw std::runtime_error("Internal error: module has already has already been initialized.");
        }

        m_exporter.push_back(exporter);
    }

private: //

private: // members
    std::vector<exporter_type> m_exporter;
    bool m_initialized = false;
};

} // namespace gridtools
} // namespace ghex
} // namespace bindings
} // namespace python
