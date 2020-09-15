/*
 * GridTools
 *
 * Copyright (c) 2014-2020, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 */
#ifndef INCLUDED_GHEX_RMA_LOCALITY_HPP
#define INCLUDED_GHEX_RMA_LOCALITY_HPP

namespace gridtools {
namespace ghex {
namespace rma {

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

template<typename Communicator>
static locality is_local(Communicator comm, int remote_rank)
{
    if (comm.rank() == remote_rank) return locality::thread;
#ifdef GHEX_USE_XPMEM
    else if (comm.is_local(remote_rank)) return locality::process;
#endif /* GHEX_USE_XPMEM */
    else return locality::remote;
}

} // namespace rma
} // namespace ghex
} // namespace gridtools

#endif /* INCLUDED_GHEX_RMA_LOCALITY_HPP */
