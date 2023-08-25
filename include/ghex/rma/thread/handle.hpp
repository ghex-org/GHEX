/*
 * ghex-org
 *
 * Copyright (c) 2014-2023, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#include <ghex/rma/locality.hpp>

namespace ghex
{
namespace rma
{
namespace thread
{
// Below are implementations of a handle in a multi-threading setting.
// Please refer to the documentation in rma/handle.hpp for further explanations.

struct info
{
    void* m_ptr;
};

struct local_data_holder
{
    void* m_ptr;

    local_data_holder(void* ptr, unsigned int, bool)
    : m_ptr{ptr}
    {
    }

    ~local_data_holder() {}

    info get_info() const { return {m_ptr}; }
};

struct remote_data_holder
{
    void* m_ptr;

    remote_data_holder(const info& info_, locality, int)
    : m_ptr{info_.m_ptr}
    {
    }

    ~remote_data_holder() {}

    void* get_ptr() const { return m_ptr; }
};

} // namespace thread
} // namespace rma
} // namespace ghex
