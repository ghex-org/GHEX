/*
 * GridTools
 *
 * Copyright (c) 2014-2019, ETH Zurich
 * All rights reserved.
 *
 * Please, refer to the LICENSE file in the root directory.
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <iostream>
#include <prototype/range_loops.hpp>
#include <prototype/halo_range.hpp>
#include <tuple>

namespace gt = gridtools;

int main() {

    auto ranges = std::make_tuple(gt::halo_range{2, 4}, gt::halo_range{4, 6});

    gt::range_loop(ranges, [](auto const& indices) { std::cout << indices[0] << ", " << indices[1] << "\n"; });

}
