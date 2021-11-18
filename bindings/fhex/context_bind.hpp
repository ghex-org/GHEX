/*
 * GridTools
 *
 * Copyright (c) 2014-2021, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <ghex/context.hpp>
#include <oomph/barrier.hpp>

namespace fhex
{
ghex::context&  context();
oomph::barrier& barrier();
} // namespace fhex
