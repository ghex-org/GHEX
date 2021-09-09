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

#include "./obj_wrapper.hpp"
#include <ghex/context.hpp>

//#include <vector>
//#include <ghex/transport_layer/util/barrier.hpp>
//
//#ifdef GHEX_USE_UCP
///* UCX backend */
//#include <ghex/transport_layer/ucx/context.hpp>
//#else
///* MPI backend */
//#include <ghex/transport_layer/mpi/context.hpp>
//#endif

namespace ghex
{
namespace fhex
{
using context_uptr_type = std::unique_ptr<ghex::context>;

extern context_uptr_type ghex_context;
//extern int               ghex_nthreads;
//extern gridtools::ghex::tl::barrier_t *ghex_barrier;
} // namespace fhex
} // namespace ghex
