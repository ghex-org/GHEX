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
#include <ghex/threads/none/primitives.hpp>
#include <vector>
#include <iomanip>
#include <utility>
#include <unistd.h>
#include <limits.h>
#include <cstring>

#include <gtest/gtest.h>

#ifdef GHEX_TEST_USE_UCX
#include <ghex/transport_layer/ucx/context.hpp>
using transport = gridtools::ghex::tl::ucx_tag;
#else
#include <ghex/transport_layer/mpi/context.hpp>
using transport = gridtools::ghex::tl::mpi_tag;
#endif

using threading = gridtools::ghex::threads::none::primitives;
using context_type = gridtools::ghex::tl::context<transport, threading>;

// test locality by collecting all local ranks
TEST(locality, enumerate) {
    auto context_ptr = gridtools::ghex::tl::context_factory<transport,threading>::create(1, MPI_COMM_WORLD);
    auto& context = *context_ptr;
    auto comm = context.get_communicator(context.get_token());

    // test self
    EXPECT_TRUE( comm.is_local(comm.rank()) );

    // check for symmetry
    std::vector<int> local_ranks(comm.size());
    // host names must be contained in a message-compatible data
    std::vector<char> my_host_name(HOST_NAME_MAX+1,0);
    std::vector<char> other_host_name(HOST_NAME_MAX+1,0);
    gethostname(my_host_name.data(), HOST_NAME_MAX+1);
    for (int r=0; r<comm.size(); ++r) {
        if (r==comm.rank()) {
            for (int rr=0; rr<comm.size(); ++rr) {
                local_ranks[rr] = comm.is_local(rr) ? 1 : 0;
            }
            for (int rr=0; rr<comm.size(); ++rr) {
                if (rr!=comm.rank()) {
                    comm.send(local_ranks, rr, 0).wait();
                    comm.send(my_host_name, rr, 1).wait();
                }
            }
        }
        else {
            const int is_neighbor = comm.is_local(r) ? 1 : 0;
            comm.recv(local_ranks, r, 0).wait();
            comm.recv(other_host_name, r, 1).wait();
            EXPECT_EQ(is_neighbor, local_ranks[comm.rank()]);
            for (int rr=0; rr<comm.size(); ++rr) {
                EXPECT_EQ((comm.is_local(rr) ? 1 : 0), local_ranks[rr]);
            }
            const int equal_hosts = (std::strcmp(my_host_name.data(), other_host_name.data()) == 0) ? 1 : 0;
            if (is_neighbor == 1)
                EXPECT_EQ(equal_hosts, 1);
        }
    }
}
