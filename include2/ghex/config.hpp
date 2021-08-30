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

#include <oomph/config.hpp>
#include <tuple>

namespace ghex
{
struct cpu
{
};
struct gpu
{
};

#if HWMALLOC_ENABLE_DEVICE
using arch_list = std::tuple<cpu, gpu>;
#else
using arch_list = std::tuple<cpu>;
#endif

} // namespace ghex