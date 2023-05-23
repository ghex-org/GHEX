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

namespace ghex
{
namespace rma
{
// source/destination of a communication
// can be either
// - among threads of the same rank,
// - among ranks on the same shared memory region
// - or remote
enum class locality
{
    thread,
    process,
    remote
};

/** @brief Determine the locality of a neighbor rank
  * @tparam Communicator Communicator type
  * @param comm a communicator instance
  * @param remote_rank neighbor rank
  * @return thread if on the same rank, process if on shared memory (provided xpmem is available)
  * and remote otherwise. */
#ifdef GHEX_NO_RMA
template<typename Communicator>
static locality
is_local(Communicator&, int)
{
    return locality::remote;
}
#else
template<typename Communicator>
static locality
is_local(Communicator& comm, int remote_rank)
{
    if (comm.rank() == remote_rank) return locality::thread;
#ifdef GHEX_USE_XPMEM
    else if (comm.is_local(remote_rank))
        return locality::process;
#endif /* GHEX_USE_XPMEM */
    else
        return locality::remote;
}
#endif

} // namespace rma
} // namespace ghex
