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
#include <ghex/context.hpp>

#ifdef GHEX_ENABLE_BARRIER
#include <ghex/barrier.hpp>
#endif

namespace fhex
{
ghex::context& context();

#if OOMPH_ENABLE_BARRIER
ghex::barrier& barrier();
#endif
} // namespace fhex
