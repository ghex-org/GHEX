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

#include <ghex/config.hpp>
#include <ghex/device/id.hpp>
#include <oomph/message_buffer.hpp>

namespace ghex
{
namespace device
{
struct guard
{
    using T = unsigned char;
    using message = oomph::message_buffer<T>;

    T*  m_ptr = nullptr;
    int m_new_device_id = 0;
    int m_current_device_id = 0;

    guard(message& m)
    {
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
        if (m.on_device())
        {
            m_ptr = m.device_data();
            m_new_device_id = m.device_id();
            m_current_device_id = get_id();
            if (m_current_device_id != m_new_device_id) set_id(m_new_device_id);
        }
        else
#endif
        {
            m_ptr = m.data();
        }
    }

    guard(guard const&) = delete;
    guard(guard&&) = delete;

    ~guard()
    {
#if defined(GHEX_USE_GPU) || defined(GHEX_GPU_MODE_EMULATE)
        if (m_new_device_id != m_current_device_id) set_id(m_current_device_id);
#endif
    }

    T* data() const noexcept { return m_ptr; }
};

} // namespace device
} // namespace ghex
