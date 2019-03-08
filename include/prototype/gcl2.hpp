/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <atomic>
#include <mpich>
#include <iostream>

std::atomic<int> counter;

class gcl_object {
    int tag_generator; // unique value to generate tags
    MPI_Comm comm; // unique communicator associated to this gcl object

    gcl_object()
        : tag_generator{counter.fetch_add(1)}
    {
        MPI_Comm_dup(MPI_COMM_WORLD, comm);
        std::cout << tag_generator << "\n";
    }
};

int main() {
    MPI_init();

    gcl_object a, b, c;
}
