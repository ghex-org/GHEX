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

namespace ghex
{
namespace rma
{
// read and write access to shared memory resource
// is either with the local (owning) thread/process
// or the remote.
enum class access_mode : int
{
    local,
    remote
};

} // namespace rma
} // namespace ghex
