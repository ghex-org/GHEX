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
#include <hwmalloc/device.hpp>

namespace ghex
{
namespace device
{
inline int
get_num()
{
    return hwmalloc::get_num_devices();
}

inline int
get_id()
{
    return hwmalloc::get_device_id();
}

inline void
set_id(int id)
{
    hwmalloc::set_device_id(id);
}

} // namespace device
} // namespace ghex
